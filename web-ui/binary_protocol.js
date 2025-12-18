(function () {
  const MAGIC = 0x31505256; // "VRP1" little-endian
  const PACKET_SIZE = 84;
  const HEADER_SIZE = 12;
  const BLOCK_SIZE = 36;
  const LEFT_OFFSET = HEADER_SIZE;
  const RIGHT_OFFSET = HEADER_SIZE + BLOCK_SIZE;
  const BUFFER_POOL_SIZE = 2;
  const BUFFER_POOL = Array.from({ length: BUFFER_POOL_SIZE }, () => new ArrayBuffer(PACKET_SIZE));
  const VIEW_POOL = BUFFER_POOL.map((buffer) => new DataView(buffer));
  let poolIndex = 0;

  function writeController(view, offset, state) {
    if (!state) {
      for (let i = 0; i < BLOCK_SIZE; i += 4) {
        view.setUint32(offset + i, 0, true);
      }
      return;
    }
    const pos = state.position || {};
    const quat = state.quaternion || {};
    view.setFloat32(offset + 0, pos.x || 0, true);
    view.setFloat32(offset + 4, pos.y || 0, true);
    view.setFloat32(offset + 8, pos.z || 0, true);
    view.setFloat32(offset + 12, quat.x || 0, true);
    view.setFloat32(offset + 16, quat.y || 0, true);
    view.setFloat32(offset + 20, quat.z || 0, true);
    view.setFloat32(offset + 24, quat.w == null ? 1 : quat.w, true);
    view.setFloat32(offset + 28, state.trigger || 0, true);
    view.setUint8(offset + 32, state.gripActive ? 1 : 0);
    view.setUint8(offset + 33, state.menuPressed ? 1 : 0);
    view.setUint16(offset + 34, 0, true);
  }

  function encodeFrame(leftState, rightState, clientTsMs, clientDtMs) {
    const buffer = BUFFER_POOL[poolIndex];
    const view = VIEW_POOL[poolIndex];
    poolIndex = (poolIndex + 1) % BUFFER_POOL_SIZE;

    view.setUint32(0, MAGIC, true);
    view.setUint32(4, clientTsMs >>> 0, true);
    const dt = Math.max(0, Math.min(65535, clientDtMs || 0));
    view.setUint16(8, dt, true);

    let flags = 0;
    if (leftState) flags |= 1;
    if (rightState) flags |= 2;
    view.setUint8(10, flags);
    view.setUint8(11, 0);

    writeController(view, LEFT_OFFSET, leftState);
    writeController(view, RIGHT_OFFSET, rightState);
    return buffer;
  }

  window.BinaryProtocol = {
    MAGIC,
    PACKET_SIZE,
    encodeFrame,
  };
})();
