(function () {
  const DEFAULT_PORT = 8442;
  const logLines = [];
  const MAX_LINES = 120;

  function appendLog(text) {
    logLines.push(`${new Date().toLocaleTimeString()}  ${text}`);
    if (logLines.length > MAX_LINES) {
      logLines.splice(0, logLines.length - MAX_LINES);
    }
    document.getElementById('log').textContent = logLines.join('\n');
  }

  function updateStatus(label, tone) {
    const statusEl = document.getElementById('status');
    statusEl.textContent = label;
    statusEl.classList.remove('status--connected', 'status--connecting');
    if (tone) {
      statusEl.classList.add(tone);
    }
  }

  document.addEventListener('DOMContentLoaded', () => {
    const wsInput = document.getElementById('wsUrl');
    const connectBtn = document.getElementById('connectBtn');
    const startBtn = document.getElementById('startBtn');
    const sceneNode = document.getElementById('vrScene');
    const handModeButtons = Array.from(document.querySelectorAll('[data-hand-mode]'));

    let sceneReady = false;
    let sceneEl = null;
    let trackingActive = false;
    let xrSession = null;
    let desiredHandMode = 'both';
    let lastLoggedHandMode = null;

    const handModeLabel = {
      both: '双手柄',
      left: '仅左手柄',
      right: '仅右手柄',
    };

    handModeButtons.forEach((btn) => {
      if (btn.classList.contains('hand-toggle__btn--active') && btn.dataset.handMode) {
        desiredHandMode = btn.dataset.handMode;
      }
    });

    const DEFAULT_HOST = '172.31.4.113';
    const host = window.location.hostname;
    const wsHost = host && host !== 'localhost' && host !== '127.0.0.1' ? host : DEFAULT_HOST;
    wsInput.value = `ws://${wsHost}:${DEFAULT_PORT}`;

    const updateTrackingButton = () => {
      startBtn.textContent = trackingActive ? '停止手柄追踪' : '开启手柄追踪';
    };

    const setTrackingState = (active) => {
      const next = Boolean(active);
      if (trackingActive === next) {
        return;
      }
      trackingActive = next;
      updateTrackingButton();
    };

    const syncHandButtons = (mode) => {
      handModeButtons.forEach((btn) => {
        const targetMode = btn.dataset.handMode;
        const isActive = targetMode === mode;
        btn.classList.toggle('hand-toggle__btn--active', isActive);
        btn.setAttribute('aria-pressed', isActive ? 'true' : 'false');
      });
    };

    const applyHandMode = (mode, logChange = true) => {
      if (mode !== 'left' && mode !== 'right' && mode !== 'both') {
        mode = 'both';
      }

      desiredHandMode = mode;

      if (sceneReady && sceneEl) {
        sceneEl.setAttribute('controller-stream', 'hands', mode);
      }

      if (logChange && mode !== lastLoggedHandMode) {
        appendLog(`🎮 姿态追踪模式：${handModeLabel[mode]}`);
        lastLoggedHandMode = mode;
      }

      syncHandButtons(mode);
    };

    if (handModeButtons.length) {
      handModeButtons.forEach((btn) => {
        btn.addEventListener('click', () => {
          applyHandMode(btn.dataset.handMode, true);
        });
      });
    }

    applyHandMode(desiredHandMode, false);

    const stopTracking = async (origin = 'button') => {
      if (!trackingActive && !xrSession) {
        appendLog('ℹ️ 手柄追踪已停止');
        return;
      }

      try {
        if (xrSession && typeof xrSession.end === 'function') {
          await xrSession.end();
          appendLog(origin === 'controller' ? '🛑 手柄请求停止手柄追踪' : '🛑 XR Session 已结束');
          xrSession = null;
        } else if (sceneEl && typeof sceneEl.exitVR === 'function' && (typeof sceneEl.is !== 'function' || sceneEl.is('vr-mode'))) {
          await sceneEl.exitVR();
          appendLog(origin === 'controller' ? '🛑 手柄请求退出 VR 会话' : '🛑 正在退出 VR 会话');
        } else {
          appendLog('ℹ️ 手柄追踪标记为停止');
        }
      } catch (error) {
        appendLog(`❌ 无法退出 VR 模式：${error.message}`);
      } finally {
        setTrackingState(false);
      }
    };

    const startTracking = async () => {
      if (!sceneReady || !sceneEl) {
        appendLog('⌛ 场景尚未初始化，请稍后再试');
        return;
      }

      if (trackingActive) {
        appendLog('ℹ️ 手柄追踪已开启');
        return;
      }

      if (typeof sceneEl.enterVR !== 'function') {
        appendLog('⚠️ 当前浏览器不支持 WebXR enterVR()');
        return;
      }

      if (typeof sceneEl.is === 'function' && sceneEl.is('vr-mode')) {
        appendLog('ℹ️ 已在 VR 模式中');
        setTrackingState(true);
        return;
      }

      try {
        await sceneEl.enterVR();
        appendLog('🎯 请求进入 VR/AR 会话');
      } catch (error) {
        appendLog(`❌ 无法进入 VR 模式：${error.message}`);
      }
    };

    const markSceneReady = (el) => {
      sceneReady = true;
      sceneEl = el;
      appendLog('✅ VR 场景已初始化');
      setTrackingState(typeof sceneEl?.is === 'function' ? sceneEl.is('vr-mode') : false);
      applyHandMode(desiredHandMode, true);

      sceneEl.addEventListener('enter-vr', () => {
        setTrackingState(true);
        appendLog('✅ VR 会话已建立');
        xrSession = sceneEl.renderer?.xr?.getSession?.() || null;
      });

      sceneEl.addEventListener('exit-vr', () => {
        setTrackingState(false);
        appendLog('🛑 VR 会话已退出');
        xrSession = null;
        if (sceneEl?.renderer && sceneEl.renderer.xr) {
          sceneEl.renderer.xr.setSession(null);
        }
      });
    };

    if (sceneNode) {
      const candidate = sceneNode.sceneEl || sceneNode;
      if (candidate && candidate.hasLoaded) {
        markSceneReady(candidate);
      } else {
        sceneNode.addEventListener(
          'loaded',
          () => {
            const el = sceneNode.sceneEl || sceneNode;
            markSceneReady(el);
          },
          { once: true }
        );
      }

      // 部分浏览器不会触发 loaded 事件，轮询 hasLoaded 兜底。
      const pollId = window.setInterval(() => {
        if (sceneReady) {
          window.clearInterval(pollId);
          return;
        }
        const el = sceneNode.sceneEl || sceneNode;
        if (el && el.hasLoaded) {
          window.clearInterval(pollId);
          markSceneReady(el);
        }
      }, 200);
    }

    connectBtn.addEventListener('click', () => {
      const url = wsInput.value.trim();
      if (!url) {
        appendLog('⚠️  请输入 WebSocket 地址');
        return;
      }
      window.VRBridge.connect(url);
    });

    startBtn.addEventListener('click', () => {
      if (!trackingActive) {
        startTracking();
      } else {
        stopTracking('button');
      }
    });

    document.addEventListener('vrbridge-stop-request', () => {
      stopTracking('controller');
    });

    updateTrackingButton();

    document.addEventListener('vrbridge-status', (event) => {
      const { status, tone } = event.detail;
      updateStatus(status, tone);
    });

    document.addEventListener('vrbridge-log', (event) => {
      appendLog(event.detail.message);
    });
  });
})();
