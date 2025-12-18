(function () {
  const EVENT_STATUS = 'vrbridge-status';
  const EVENT_LOG = 'vrbridge-log';

  const WebRTCBridge = {
    signaling: null,
    peer: null,
    channel: null,
    url: '',
    ready: false,
    channelName: 'controller',
    reconnectDelayMs: 1500,
    reconnectTimer: null,
    pendingCandidates: [],
    shouldReconnect: false,
    suppressChannelClose: false, // 避免手动关闭时触发重连逻辑
    channelCloseHandler: null,

    connect(url) {
      this.disconnect();
      this.url = url;
      this.shouldReconnect = true;
      this._status('连接中…', 'status--connecting');
      this._log(`🔌 尝试连接 ${url}`);

      if (typeof window.RTCPeerConnection !== 'function') {
        this._log('❌ 当前浏览器不支持 WebRTC');
        this._status('未连接');
        return;
      }

      try {
        const ws = new WebSocket(url);
        this.signaling = ws;

        ws.addEventListener('open', () => {
          this._log('✅ 信令通道已建立');
          this._createPeer();
        });

        ws.addEventListener('message', (event) => {
          this._handleSignal(event.data);
        });

        ws.addEventListener('close', (event) => {
          const reason = event.reason || '信令通道关闭';
          this._log(`⚠️ WebSocket 信令关闭：${reason}`);
          this._status('未连接');
          this.ready = false;
          this._disposePeer();
          this._emitStop('signaling-close');
        });

        ws.addEventListener('error', (error) => {
          this._log(`❌ WebSocket 信令错误：${error.message || error}`);
        });
      } catch (error) {
        this._log(`❌ 无法创建信令连接：${error.message}`);
        this._status('未连接');
      }
    },

    disconnect() {
      this.shouldReconnect = false;
      this._clearReconnect();

      if (this.signaling && this.signaling.readyState === WebSocket.OPEN) {
        try {
          this.signaling.send(JSON.stringify({ type: 'bye' }));
        } catch (_) {
          /* noop */
        }
      }

      if (this.channel) {
        if (this.channelCloseHandler) {
          this.channel.removeEventListener('close', this.channelCloseHandler);
          this.channelCloseHandler = null;
        }
        this.suppressChannelClose = true;
        try {
          this.channel.close();
        } catch (_) {
          /* noop */
        }
      }
      this.channel = null;

      if (this.suppressChannelClose) {
        window.setTimeout(() => {
          this.suppressChannelClose = false;
        }, 0);
      }

      if (this.peer) {
        try {
          this.peer.close();
        } catch (_) {
          /* noop */
        }
      }
      this.peer = null;

      if (this.signaling) {
        try {
          this.signaling.close();
        } catch (_) {
          /* noop */
        }
      }
      this.signaling = null;

      this.ready = false;
      this.pendingCandidates = [];
      this._status('未连接');
    },

    send(payload) {
      if (!this.channel || this.channel.readyState !== 'open') {
        return;
      }
      try {
        this.channel.send(JSON.stringify(payload));
      } catch (error) {
        this._log(`❌ DataChannel 发送失败：${error.message}`);
      }
    },

    _createPeer() {
      if (!this.signaling || this.signaling.readyState !== WebSocket.OPEN) {
        this._log('⚠️ 信令通道未就绪，无法建立 PeerConnection');
        return;
      }

      this._disposePeer();
      this.peer = new RTCPeerConnection();
      this.pendingCandidates = [];

      this.peer.addEventListener('icecandidate', (event) => {
        if (!event.candidate) {
          return;
        }
        this._sendSignal({
          type: 'ice',
          candidate: {
            candidate: event.candidate.candidate,
            sdpMid: event.candidate.sdpMid,
            sdpMLineIndex: event.candidate.sdpMLineIndex,
          },
        });
      });

      this.peer.addEventListener('connectionstatechange', () => {
        const state = this.peer?.connectionState;
        if (state) {
          this._log(`ℹ️ 连接状态：${state}`);
        }
        if (state === 'failed' || state === 'disconnected') {
          this.ready = false;
          this._status('连接中…', 'status--connecting');
          this._emitStop('connection-state');
          this._restartPeer();
        }
      });

      const channel = this.peer.createDataChannel(this.channelName);
      this._attachChannel(channel);

      this._negotiate();
    },

    async _negotiate() {
      if (!this.peer || !this.signaling || this.signaling.readyState !== WebSocket.OPEN) {
        return;
      }

      try {
        const offer = await this.peer.createOffer();
        await this.peer.setLocalDescription(offer);
        const local = this.peer.localDescription;
        if (local) {
          this._sendSignal({ type: local.type, sdp: local.sdp });
          this._log('📤 已发送 WebRTC offer');
        }
      } catch (error) {
        this._log(`❌ 创建 offer 失败：${error.message}`);
      }
    },

    async _handleSignal(raw) {
      let payload;
      try {
        payload = typeof raw === 'string' ? JSON.parse(raw) : raw;
      } catch (error) {
        this._log('⚠️ 无法解析信令消息');
        return;
      }

      const { type } = payload || {};
      if (!type) {
        return;
      }

      if (type === 'answer') {
        await this._handleAnswer(payload);
        return;
      }

      if (type === 'ice') {
        await this._handleRemoteCandidate(payload);
        return;
      }

      if (type === 'error') {
        this._log(`❌ 信令错误：${payload.reason || '未知原因'}`);
        return;
      }

      this._log(`⚠️ 未知信令类型：${type}`);
    },

    async _handleAnswer(payload) {
      if (!this.peer) {
        this._log('⚠️ 收到 answer 时 PeerConnection 不存在');
        return;
      }

      const { sdp } = payload;
      if (!sdp) {
        this._log('⚠️ answer 缺少 SDP');
        return;
      }

      try {
        await this.peer.setRemoteDescription({ type: 'answer', sdp });
        this._log('📥 已接收 WebRTC answer');
        await this._flushPendingCandidates();
      } catch (error) {
        this._log(`❌ 设置远端描述失败：${error.message}`);
      }
    },

    async _handleRemoteCandidate(payload) {
      if (!this.peer) {
        return;
      }

      const candidatePayload = payload.candidate;
      if (!candidatePayload) {
        if (payload.endOfCandidates && typeof this.peer.addIceCandidate === 'function') {
          try {
            await this.peer.addIceCandidate(null);
          } catch (error) {
            this._log(`⚠️ 结束 ICE 时出错：${error.message}`);
          }
        }
        return;
      }

      const candidate = new RTCIceCandidate(candidatePayload);
      if (!this.peer.remoteDescription) {
        this.pendingCandidates.push(candidate);
        return;
      }

      try {
        await this.peer.addIceCandidate(candidate);
      } catch (error) {
        this._log(`⚠️ 添加远端 ICE 失败：${error.message}`);
      }
    },

    async _flushPendingCandidates() {
      if (!this.peer || !this.peer.remoteDescription) {
        return;
      }

      const queued = this.pendingCandidates;
      this.pendingCandidates = [];
      for (const candidate of queued) {
        try {
          await this.peer.addIceCandidate(candidate);
        } catch (error) {
          this._log(`⚠️ 添加缓存 ICE 失败：${error.message}`);
        }
      }
    },

    _attachChannel(channel) {
      this.channel = channel;

      channel.addEventListener('open', () => {
        this.ready = true;
        this._status('已连接', 'status--connected');
        this._log('✅ DataChannel 已打开');
      });

      const handleClose = () => {
        this.ready = false;
        if (this.suppressChannelClose) {
          return;
        }
        this._status('连接中…');
        this._log('⚠️ DataChannel 已关闭');
        this._emitStop('channel-close');
        this._restartPeer();
      };

      channel.addEventListener('close', handleClose);
      this.channelCloseHandler = handleClose;

      channel.addEventListener('error', (event) => {
        const message = event?.message || event?.error?.message;
        this._log(`⚠️ DataChannel 错误：${message || '未知错误'}`);
      });

      channel.addEventListener('message', (event) => {
        if (event?.data) {
          this._log(`ℹ️ 收到 DataChannel 消息：${event.data}`);
        }
      });
    },

    _restartPeer() {
      if (!this.shouldReconnect) {
        return;
      }
      if (!this.signaling || this.signaling.readyState !== WebSocket.OPEN) {
        return;
      }

      this._clearReconnect();
      this.reconnectTimer = window.setTimeout(() => {
        this.reconnectTimer = null;
        this._log('🔁 重新协商 DataChannel');
        this._createPeer();
      }, this.reconnectDelayMs);
    },

    _disposePeer() {
      if (this.channel) {
        if (this.channelCloseHandler) {
          this.channel.removeEventListener('close', this.channelCloseHandler);
          this.channelCloseHandler = null;
        }
        this.suppressChannelClose = true;
        try {
          this.channel.close();
        } catch (_) {
          /* noop */
        }
      }
      this.channel = null;

      if (this.suppressChannelClose) {
        window.setTimeout(() => {
          this.suppressChannelClose = false;
        }, 0);
      }

      if (this.peer) {
        try {
          this.peer.close();
        } catch (_) {
          /* noop */
        }
      }
      this.peer = null;
      this.ready = false;
    },

    _clearReconnect() {
      if (this.reconnectTimer) {
        window.clearTimeout(this.reconnectTimer);
        this.reconnectTimer = null;
      }
    },

    _sendSignal(message) {
      if (!this.signaling || this.signaling.readyState !== WebSocket.OPEN) {
        return;
      }
      try {
        this.signaling.send(JSON.stringify(message));
      } catch (error) {
        this._log(`⚠️ 发送信令失败：${error.message}`);
      }
    },

    _emitStop(source) {
      document.dispatchEvent(
        new CustomEvent('vrbridge-stop-request', {
          detail: { source },
        })
      );
    },

    _status(text, tone) {
      document.dispatchEvent(
        new CustomEvent(EVENT_STATUS, {
          detail: { status: text, tone },
        })
      );
    },

    _log(message) {
      document.dispatchEvent(
        new CustomEvent(EVENT_LOG, {
          detail: { message },
        })
      );
    },
  };

  window.VRBridge = WebRTCBridge;

  AFRAME.registerComponent('controller-stream', {
    schema: {
      hands: { type: 'string', default: 'both' },
    },

    init() {
      this.scene = this.el.sceneEl || this.el;
      this._xrSession = null;
      this._refSpace = null;
      this._active = false;
      this._frameRateNegotiated = false;
      this._binaryEnabled = !new URLSearchParams(window.location.search).has('json');
      this._prevPixelRatio = null;
      this._prevFoveation = null;
      this._lastClientTs = null;
      this._onXRFrame = this._onXRFrame.bind(this);
      this._emptyButtons = [];
      this._leftState = {
        hand: 'left',
        position: { x: 0, y: 0, z: 0 },
        quaternion: { x: 0, y: 0, z: 0, w: 1 },
        gripActive: false,
        trigger: 0,
        menuPressed: false,
      };
      this._rightState = {
        hand: 'right',
        position: { x: 0, y: 0, z: 0 },
        quaternion: { x: 0, y: 0, z: 0, w: 1 },
        gripActive: false,
        trigger: 0,
        menuPressed: false,
      };

      this._handleEnterVR = () => {
        this._startXRLoop();
      };
      this._handleExitVR = () => {
        this._stopXRLoop();
      };
      if (this.scene) {
        this.scene.addEventListener('enter-vr', this._handleEnterVR);
        this.scene.addEventListener('exit-vr', this._handleExitVR);
      }
    },

    pause() {
      this._stopXRLoop();
    },

    remove() {
      if (this.scene) {
        this.scene.removeEventListener('enter-vr', this._handleEnterVR);
        this.scene.removeEventListener('exit-vr', this._handleExitVR);
      }
      this._stopXRLoop();
    },

    async _startXRLoop() {
      if (this._active || !this.scene) {
        return;
      }
      const renderer = this.scene.renderer;
      const session = renderer && renderer.xr ? renderer.xr.getSession() : null;
      if (!session) {
        return;
      }

      // 降低 VR 渲染负载，帮助 XR session 稳定高帧率。
      if (renderer && typeof renderer.getPixelRatio === 'function' && typeof renderer.setPixelRatio === 'function') {
        try {
          this._prevPixelRatio = renderer.getPixelRatio();
          // Quest 上不依赖清晰渲染，直接把像素比砍到 0.5，优先换帧率稳定。
          renderer.setPixelRatio(0.5);
        } catch (_) {
          this._prevPixelRatio = null;
        }
      }
      if (renderer && renderer.xr && typeof renderer.xr.setFoveation === 'function') {
        try {
          this._prevFoveation =
            typeof renderer.xr.getFoveation === 'function' ? renderer.xr.getFoveation() : null;
          renderer.xr.setFoveation(1);
        } catch (_) {
          this._prevFoveation = null;
        }
      }
      if (renderer && renderer.xr && typeof renderer.xr.setFramebufferScaleFactor === 'function') {
        try {
          renderer.xr.setFramebufferScaleFactor(0.5);
        } catch (_) {
          // ignore
        }
      }

      this._xrSession = session;
      try {
        this._refSpace = await session.requestReferenceSpace('local');
      } catch (_) {
        this._xrSession = null;
        this._refSpace = null;
        return;
      }
      this._active = true;
      this._frameRateNegotiated = false;
      this._lastClientTs = Date.now();

      session.addEventListener('end', () => this._stopXRLoop(), { once: true });
      session.requestAnimationFrame(this._onXRFrame);
    },

    _stopXRLoop() {
      this._active = false;
      if (this.scene && this.scene.renderer) {
        const renderer = this.scene.renderer;
        if (this._prevPixelRatio != null && typeof renderer.setPixelRatio === 'function') {
          try {
            renderer.setPixelRatio(this._prevPixelRatio);
          } catch (_) {}
        }
        if (
          this._prevFoveation != null
          && renderer.xr
          && typeof renderer.xr.setFoveation === 'function'
        ) {
          try {
            renderer.xr.setFoveation(this._prevFoveation);
          } catch (_) {}
        }
      }
      this._xrSession = null;
      this._refSpace = null;
      this._frameRateNegotiated = false;
      this._prevPixelRatio = null;
      this._prevFoveation = null;
      this._lastClientTs = null;
    },

    _getHandsMode() {
      const rawMode = this.data.hands;
      return rawMode === 'left' || rawMode === 'right' ? rawMode : 'both';
    },

    _readButtonValue(button) {
      if (!button) return 0;
      if (typeof button.value === 'number') {
        return button.value;
      }
      return button.pressed ? 1 : 0;
    },

    _onXRFrame(_time, frame) {
      if (!this._active || !frame || !this._xrSession || !this._refSpace) {
        return;
      }

      if (!this._frameRateNegotiated) {
        this._frameRateNegotiated = true;
        const session = frame.session;
        const supportedRates = session.supportedFrameRates;
        let numericRates = [];
        if (supportedRates != null) {
          try {
            numericRates = Array.from(supportedRates).filter(
              (val) => typeof val === 'number' && Number.isFinite(val),
            );
          } catch (_) {
            numericRates = [];
          }
        }
        if (numericRates.length) {
          WebRTCBridge._log(`🎯 supportedFrameRates: ${numericRates.join(', ')} Hz`);
        } else {
          WebRTCBridge._log('🎯 supportedFrameRates: unavailable');
        }

        const currentRate = session.frameRate;
        if (typeof currentRate === 'number' && Number.isFinite(currentRate)) {
          WebRTCBridge._log(`🎯 current frameRate: ${currentRate} Hz`);
        } else if (currentRate !== undefined) {
          WebRTCBridge._log(`🎯 current frameRate: ${currentRate}`);
        } else {
          WebRTCBridge._log('🎯 current frameRate: unavailable');
        }

        if (typeof session.updateTargetFrameRate === 'function') {
          if (numericRates.length) {
            // 优先请求 90Hz（Quest2 默认上限），避免盲目冲到 120 触发降帧。
            const preferredRates = [90, 72];
            let targetRate = null;
            for (const rate of preferredRates) {
              if (numericRates.includes(rate)) {
                targetRate = rate;
                break;
              }
            }
            if (targetRate == null) {
              const le90 = numericRates.filter((rate) => rate <= 90);
              targetRate = le90.length ? Math.max(...le90) : Math.max(...numericRates);
            }
            session.updateTargetFrameRate(targetRate).then(
              () => {
                WebRTCBridge._log(`🎯 updateTargetFrameRate(${targetRate}) ok`);
              },
              (err) => {
                const message = err && err.message ? err.message : String(err);
                WebRTCBridge._log(`⚠️ updateTargetFrameRate(${targetRate}) failed: ${message}`);
              },
            );
          } else {
            WebRTCBridge._log('🎯 updateTargetFrameRate supported but rates unavailable');
          }
        } else {
          WebRTCBridge._log('🎯 updateTargetFrameRate: unsupported');
        }
      }
      if (!window.VRBridge || !window.VRBridge.ready) {
        this._xrSession.requestAnimationFrame(this._onXRFrame);
        return;
      }

      // DataChannel 回压保护：如果发送队列堆积过多，直接跳过本帧，避免拖垮 XR 帧率。
      const channel = window.VRBridge.channel;
      if (channel && typeof channel.bufferedAmount === 'number') {
        const maxBufferedBytes = 256 * 1024;
        if (channel.bufferedAmount > maxBufferedBytes) {
          this._xrSession.requestAnimationFrame(this._onXRFrame);
          return;
        }
      }

      const now = Date.now();
      let clientDtMs = 0;
      if (typeof this._lastClientTs === 'number') {
        const diff = now - this._lastClientTs;
        if (diff >= 0) {
          clientDtMs = diff;
        }
      }
      this._lastClientTs = now;

      const mode = this._getHandsMode();
      const shouldSend = (handKey) => mode === 'both' || mode === handKey;

      let leftState = null;
      let rightState = null;
      const emptyButtons = this._emptyButtons;
      for (const source of frame.session.inputSources) {
        const handKey = source.handedness;
        if (handKey !== 'left' && handKey !== 'right') {
          continue;
        }
        if (!shouldSend(handKey)) {
          continue;
        }
        const space = source.gripSpace || source.targetRaySpace;
        if (!space) {
          continue;
        }
        const pose = frame.getPose(space, this._refSpace);
        if (!pose) {
          continue;
        }

        const { position, orientation } = pose.transform;
        const buttons =
          source.gamepad && Array.isArray(source.gamepad.buttons) ? source.gamepad.buttons : emptyButtons;
        const triggerValue = this._readButtonValue(buttons[0]);
        const gripValue = this._readButtonValue(buttons[1]);
        const primaryButton = buttons[4] || buttons[3];
        const menuPressed = Boolean(primaryButton && primaryButton.pressed);

        const state = handKey === 'left' ? this._leftState : this._rightState;
        const pos = state.position;
        pos.x = position.x;
        pos.y = position.y;
        pos.z = position.z;
        const quat = state.quaternion;
        quat.x = orientation.x;
        quat.y = orientation.y;
        quat.z = orientation.z;
        quat.w = orientation.w;
        state.gripActive = gripValue > 0.5;
        state.trigger = triggerValue;
        state.menuPressed = menuPressed;

        if (handKey === 'left') {
          leftState = state;
        } else {
          rightState = state;
        }
      }

      const hasData = Boolean(leftState || rightState);
      if (hasData) {
        const useBinary =
          window.BinaryProtocol
          && typeof window.BinaryProtocol.encodeFrame === 'function'
          && this._binaryEnabled;
        if (useBinary) {
          try {
            const buffer = window.BinaryProtocol.encodeFrame(leftState, rightState, now, clientDtMs);
            if (channel && channel.readyState === 'open') {
              channel.send(buffer);
            }
          } catch (err) {
            // eslint-disable-next-line no-console
            console.warn('Failed to send binary controller state', err);
          }
        } else {
          const payload = { timestamp: now, debug_version: "ts_v2" };
          payload.client_ts = now;
          payload.client_dt = clientDtMs;
          if (leftState) payload.leftController = leftState;
          if (rightState) payload.rightController = rightState;
          try {
            window.VRBridge.send(payload);
          } catch (err) {
            // eslint-disable-next-line no-console
            console.warn('Failed to send controller state', err);
          }
        }
      }

      this._xrSession.requestAnimationFrame(this._onXRFrame);
    },

    update(oldData) {
      if (!oldData) return;
    },
  });
})();
