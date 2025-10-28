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

  function captureGamepadState(controllerEl) {
    const tracked = controllerEl.components['tracked-controls']?.controller;
    const gamepad = tracked?.gamepad;
    if (!gamepad) {
      return { gripActive: false, trigger: 0, menuPressed: false };
    }

    const buttons = Array.isArray(gamepad.buttons) ? gamepad.buttons : [];
    const triggerButton = buttons[0];
    const gripButton = buttons[1];

    const readValue = (button) => {
      if (!button) return 0;
      if (typeof button.value === 'number') {
        return button.value;
      }
      return button.pressed ? 1 : 0;
    };

    const triggerValue = readValue(triggerButton);
    const gripValue = readValue(gripButton);
    const menuPressed = false;
    return {
      gripActive: gripValue > 0.5,
      trigger: triggerValue,
      menuPressed,
    };
  }

  AFRAME.registerComponent('controller-stream', {
    schema: {
      interval: { type: 'number', default: 20 },
      scale: { type: 'number', default: 1.0 },
      hands: { type: 'string', default: 'both' },
    },

    init() {
      this.left = document.getElementById('leftController');
      this.right = document.getElementById('rightController');
      this.lastSent = 0;
      this.scene = this.el.sceneEl || this.el;
    },

    tick(time, delta) {
      if (!window.VRBridge || !window.VRBridge.ready) {
        return;
      }

      if (!this.scene) {
        return;
      }

      if (typeof this.scene.is === 'function' && !this.scene.is('vr-mode')) {
        return;
      }

      this.lastSent += delta;
      if (this.lastSent < this.data.interval) {
        return;
      }
      this.lastSent = Math.max(0, this.lastSent - this.data.interval);

      const payload = { timestamp: Date.now() };
      let hasData = false;

      const rawMode = this.data.hands;
      const mode = rawMode === 'left' || rawMode === 'right' ? rawMode : 'both';
      const shouldSend = (handKey) => {
        if (mode === 'both') {
          return true;
        }
        return mode === handKey;
      };

      const processController = (controllerEl, handKey) => {
        if (!controllerEl || !controllerEl.object3D.visible) {
          return null;
        }

        const pos = controllerEl.object3D.position;
        const quat = controllerEl.object3D.quaternion;
        const buttons = captureGamepadState(controllerEl);

        return {
          hand: handKey,
          position: { x: pos.x, y: pos.y, z: pos.z },
          quaternion: { x: quat.x, y: quat.y, z: quat.z, w: quat.w },
          gripActive: buttons.gripActive,
          trigger: buttons.trigger,
          menuPressed: buttons.menuPressed,
        };
      };

      const leftState = shouldSend('left') ? processController(this.left, 'left') : null;
      const rightState = shouldSend('right') ? processController(this.right, 'right') : null;

      if (leftState) {
        payload.leftController = leftState;
        hasData = true;
      }
      if (rightState) {
        payload.rightController = rightState;
        hasData = true;
      }

      if (hasData) {
        window.VRBridge.send(payload);
      }

    },
  });
})();
