"""WebRTC 信令与 DataChannel 服务器实现。"""

from __future__ import annotations

import asyncio
import json
import logging
from asyncio import AbstractServer
from typing import Any, Dict, Optional, Sequence, TYPE_CHECKING

import websockets
from websockets.exceptions import ConnectionClosed
from websockets.server import WebSocketServerProtocol

from aiortc import (
    RTCConfiguration,
    RTCDataChannel,
    RTCIceCandidate,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
)

if TYPE_CHECKING:  # pragma: no cover - 仅用于类型提示
    from .controller_pipeline import ControllerPipeline

logger = logging.getLogger(__name__)


class VRWebRTCServer:
    """管理单客户端 WebRTC 会话，通过 DataChannel 转发手柄数据。"""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        pipeline: "ControllerPipeline",
        channel_name: str = "controller",
        stun_servers: Optional[Sequence[str]] = None,
    ) -> None:
        self.host = host
        self.port = port
        self.pipeline = pipeline
        self.channel_name = channel_name
        self.stun_servers = list(stun_servers or [])

        self._server: Optional[AbstractServer] = None
        self._websocket: Optional[WebSocketServerProtocol] = None
        self._peer: Optional[RTCPeerConnection] = None
        self._channel: Optional[RTCDataChannel] = None
        self._busy_message = json.dumps({"type": "error", "reason": "busy"})
        # 单客户端模式：通过锁避免同时受理多个信令连接。
        self._client_lock = asyncio.Lock()

    async def start(self) -> None:
        if self._server is not None:
            raise RuntimeError("Server already started")

        self._server = await websockets.serve(self._handle_signaling, self.host, self.port)
        if self.pipeline.allowed_hands == {"left", "right"}:
            hands_label = "双手柄"
        else:
            hands_label = ",".join(sorted(self.pipeline.allowed_hands))
        logger.info(
            "WebRTC signaling server listening on ws://%s:%s (hands: %s, channel: %s)",
            self.host,
            self.port,
            hands_label,
            self.channel_name,
        )

    async def stop(self) -> None:
        await self._cleanup_peer()
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        logger.info("WebRTC signaling server stopped")

    async def _handle_signaling(self, websocket: WebSocketServerProtocol) -> None:
        async with self._client_lock:
            if self._websocket is not None:
                await websocket.send(self._busy_message)
                await websocket.close()
                return
            self._websocket = websocket

        logger.info("Signaling client connected: %s", websocket.remote_address)
        try:
            async for message in websocket:
                await self._dispatch_signal(message)
        except ConnectionClosed:
            logger.info("Signaling client disconnected: %s", websocket.remote_address)
        finally:
            if self._websocket is websocket:
                self._websocket = None
            await self._cleanup_peer()

    async def _dispatch_signal(self, message: str) -> None:
        try:
            payload = json.loads(message)
        except json.JSONDecodeError:
            logger.warning("Discarding invalid signaling payload: %s", message)
            return

        msg_type = payload.get("type")
        if msg_type == "offer":
            await self._accept_offer(payload.get("sdp"))
        elif msg_type == "ice":
            await self._add_remote_candidate(payload)
        elif msg_type == "bye":
            await self._cleanup_peer()
        else:
            logger.warning("Unsupported signaling message type: %s", msg_type)

    async def _accept_offer(self, sdp: Optional[str]) -> None:
        if not sdp:
            logger.warning("Offer message missing SDP")
            return

        await self._cleanup_peer()

        configuration = self._build_configuration()
        peer = RTCPeerConnection(configuration=configuration)
        self._peer = peer

        @peer.on("datachannel")
        def _on_datachannel(channel: RTCDataChannel) -> None:
            logger.info("DataChannel received: %s", channel.label)
            if channel.label != self.channel_name:
                logger.warning("Unexpected DataChannel label %s (expect %s)", channel.label, self.channel_name)
            self._attach_channel(channel)

        @peer.on("connectionstatechange")
        async def _on_connectionstatechange() -> None:
            logger.info("Peer connection state: %s", peer.connectionState)
            if peer.connectionState in {"failed", "closed"}:
                await self._cleanup_peer()

        @peer.on("icecandidate")
        async def _on_icecandidate(event) -> None:  # type: ignore[no-redef]
            if not event.candidate:
                return
            await self._send_signal(
                {
                    "type": "ice",
                    "candidate": {
                        "candidate": event.candidate.to_sdp(),
                        "sdpMid": event.candidate.sdpMid,
                        "sdpMLineIndex": event.candidate.sdpMLineIndex,
                    },
                }
            )

        try:
            await peer.setRemoteDescription(RTCSessionDescription(sdp=sdp, type="offer"))
            answer = await peer.createAnswer()
            await peer.setLocalDescription(answer)
        except Exception as exc:  # pragma: no cover - aiortc 内部错误
            logger.error("Failed to negotiate WebRTC session: %s", exc)
            await self._cleanup_peer()
            return

        if peer.localDescription:
            await self._send_signal({"type": peer.localDescription.type, "sdp": peer.localDescription.sdp})

    async def _add_remote_candidate(self, payload: Dict[str, Any]) -> None:
        if not self._peer:
            logger.warning("Received ICE candidate without active peer")
            return

        candidate_payload = payload.get("candidate")
        if candidate_payload is None and payload.get("endOfCandidates"):
            await self._peer.addIceCandidate(None)
            return

        if isinstance(candidate_payload, dict):
            candidate_str = candidate_payload.get("candidate")
            sdp_mid = candidate_payload.get("sdpMid")
            sdp_mline_index = candidate_payload.get("sdpMLineIndex")
        else:
            candidate_str = payload.get("candidate")
            sdp_mid = payload.get("sdpMid")
            sdp_mline_index = payload.get("sdpMLineIndex")

        if not candidate_str:
            logger.warning("ICE candidate payload missing candidate string")
            return

        sdp_mid = sdp_mid or self.channel_name or "data"

        try:
            candidate = RTCIceCandidate(
                sdp_mid,
                int(sdp_mline_index) if sdp_mline_index is not None else 0,
                candidate_str,
            )
        except Exception as exc:  # pragma: no cover - aiortc 参数校验
            logger.warning(
                "Invalid ICE candidate payload: %s (raw=%s)",
                exc,
                candidate_payload or payload,
            )
            return
        try:
            await self._peer.addIceCandidate(candidate)
        except Exception as exc:  # pragma: no cover - aiortc 内部错误
            logger.warning("Failed to add ICE candidate: %s", exc)

    def _attach_channel(self, channel: RTCDataChannel) -> None:
        self._channel = channel

        @channel.on("open")
        def _on_open() -> None:
            logger.info("DataChannel open (%s)", channel.label)

        @channel.on("close")
        def _on_close() -> None:
            logger.info("DataChannel closed (%s)", channel.label)
            asyncio.create_task(self._cleanup_peer())

        @channel.on("message")
        def _on_message(message) -> None:  # type: ignore[override]
            if isinstance(message, bytes):
                try:
                    message = message.decode("utf-8")
                except UnicodeDecodeError:
                    logger.warning("Discarding non-UTF8 DataChannel payload")
                    return

            if not isinstance(message, str):
                logger.warning("Unsupported DataChannel payload type: %s", type(message))
                return

            try:
                payload = json.loads(message)
            except json.JSONDecodeError:
                logger.warning("Failed to decode controller payload: %s", message)
                return

            goals = self.pipeline.process_message(payload)
            if goals:
                logger.debug("Teleop summaries: %s", goals)

    async def _cleanup_peer(self) -> None:
        peer, channel = self._peer, self._channel
        self._peer = None
        self._channel = None

        if channel:
            try:
                channel.close()
            except Exception:
                logger.debug("Ignoring channel.close() error", exc_info=True)

        if peer:
            try:
                await peer.close()
            except Exception:
                logger.debug("Ignoring peer.close() error", exc_info=True)

        self.pipeline.reset()

    def _build_configuration(self) -> Optional[RTCConfiguration]:
        if not self.stun_servers:
            return RTCConfiguration(iceServers=[])
        ice_servers = [RTCIceServer(urls=self.stun_servers)]
        return RTCConfiguration(iceServers=ice_servers)

    async def _send_signal(self, message: Dict[str, Any]) -> None:
        websocket = self._websocket
        if websocket is None:
            logger.warning("Cannot send signaling message, websocket missing")
            return

        try:
            await websocket.send(json.dumps(message))
        except Exception as exc:
            logger.warning("Failed to send signaling message: %s", exc)
