// pend.js — the film's GPU double-pendulum integrator (WebGL2, 32-bit floats). Same equations and the same classical RK4
// as sim/dp/dp.c (float64 on the CPU). One texel = one pendulum: state texture RGBA32F (th1, th2, w1, w2) and a flip
// texture (first time either arm passed over the top, -1 = not yet). Used by the film and by gpucheck.html (check V5).
// m1 = m2 = 1 kg, l1 = l2 = 1 m, g = 9.81 m/s^2 (uniforms).
const VS = `#version 300 es
in vec2 p; void main(){ gl_Position = vec4(p, 0., 1.); }`;
const STEP = `#version 300 es
precision highp float;
uniform sampler2D uS; uniform sampler2D uF; uniform float uDt; uniform int uN; uniform float uT0; uniform float uG;
layout(location=0) out vec4 oS; layout(location=1) out vec4 oF;
vec4 rhs(vec4 s){
  float t1 = s.x, t2 = s.y, w1 = s.z, w2 = s.w; float dl = t1 - t2, sd = sin(dl), cd = cos(dl);
  float den = 3. - cos(2. * dl);          // 2 m1 + m2 - m2 cos(2 dl) with m1 = m2 = 1
  float a1 = (-uG * 3. * sin(t1) - uG * sin(t1 - 2. * t2) - 2. * sd * (w2 * w2 + w1 * w1 * cd)) / den;
  float a2 = (2. * sd * (w1 * w1 * 2. + uG * 2. * cos(t1) + w2 * w2 * cd)) / den;
  return vec4(w1, w2, a1, a2);
}
void main(){
  ivec2 ij = ivec2(gl_FragCoord.xy); vec4 s = texelFetch(uS, ij, 0); vec4 f = texelFetch(uF, ij, 0);
  float h = uDt;
  for (int k = 0; k < uN; k++) {
    vec4 k1 = rhs(s); vec4 k2 = rhs(s + 0.5 * h * k1); vec4 k3 = rhs(s + 0.5 * h * k2); vec4 k4 = rhs(s + h * k3);
    s += h / 6. * (k1 + 2. * k2 + 2. * k3 + k4);
    if (f.x < 0. && (abs(s.x) > 3.14159265 || abs(s.y) > 3.14159265)) f.x = uT0 + float(k + 1) * h;
  }
  oS = s; oF = f;
}`;
function sh(gl, type, src) { const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s); if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s)); return s; }
export function program(gl, vs, fs) { const p = gl.createProgram(); gl.attachShader(p, sh(gl, gl.VERTEX_SHADER, vs)); gl.attachShader(p, sh(gl, gl.FRAGMENT_SHADER, fs)); gl.linkProgram(p); if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p)); return p; }
let QUAD = null;
export function quad(gl) { if (QUAD) return QUAD; QUAD = gl.createVertexArray(); gl.bindVertexArray(QUAD); const b = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, b); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW); gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0); gl.bindVertexArray(null); return QUAD; }
function tex(gl, w, h, data) { const t = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, t); gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, w, h, 0, gl.RGBA, gl.FLOAT, data || null);
  for (const [k, v] of [[gl.TEXTURE_MIN_FILTER, gl.NEAREST], [gl.TEXTURE_MAG_FILTER, gl.NEAREST], [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE]]) gl.texParameteri(gl.TEXTURE_2D, k, v); return t; }

export class PendGPU {
  // init: Float32Array of w*h*4 (th1, th2, w1, w2); row-major, texel (i % w, floor(i / w)) = pendulum i
  constructor(gl, w, h, init, o = {}) {
    if (!gl.getExtension('EXT_color_buffer_float')) throw new Error('EXT_color_buffer_float missing');
    this.gl = gl; this.w = w; this.h = h; this.dt = o.dt ?? 1 / 1200; this.g = o.g ?? 9.81; this.k = 0; this.init = init;
    this.prog = this.prog || program(gl, VS, STEP);
    const fl = new Float32Array(w * h * 4); for (let i = 0; i < w * h; i++) fl[4 * i] = -1;
    this.flipInit = fl;
    this.S = [tex(gl, w, h, init), tex(gl, w, h)]; this.F = [tex(gl, w, h, fl), tex(gl, w, h)]; this.cur = 0;
    this.fb = [0, 1].map(i => { const f = gl.createFramebuffer(); gl.bindFramebuffer(gl.FRAMEBUFFER, f);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.S[i], 0); gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.TEXTURE_2D, this.F[i], 0);
      gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1]); return f; });
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }
  reset() { const gl = this.gl; gl.bindTexture(gl.TEXTURE_2D, this.S[0]); gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, this.w, this.h, gl.RGBA, gl.FLOAT, this.init);
    gl.bindTexture(gl.TEXTURE_2D, this.F[0]); gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, this.w, this.h, gl.RGBA, gl.FLOAT, this.flipInit); this.cur = 0; this.k = 0; }
  get t() { return this.k * this.dt; }
  // advance to step index K (resets if K is behind); at most `chunk` RK4 steps per draw call
  advanceTo(K, chunk = 64) {
    const gl = this.gl; if (K < this.k) this.reset();
    gl.useProgram(this.prog); gl.bindVertexArray(quad(gl)); gl.viewport(0, 0, this.w, this.h); gl.disable(gl.BLEND);
    while (this.k < K) {
      const n = Math.min(chunk, K - this.k), src = this.cur, dst = 1 - src;
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.fb[dst]);
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.S[src]); gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.F[src]);
      gl.uniform1i(gl.getUniformLocation(this.prog, 'uS'), 0); gl.uniform1i(gl.getUniformLocation(this.prog, 'uF'), 1);
      gl.uniform1f(gl.getUniformLocation(this.prog, 'uDt'), this.dt); gl.uniform1i(gl.getUniformLocation(this.prog, 'uN'), n);
      gl.uniform1f(gl.getUniformLocation(this.prog, 'uT0'), this.k * this.dt); gl.uniform1f(gl.getUniformLocation(this.prog, 'uG'), this.g);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4); this.cur = dst; this.k += n;
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null); gl.bindVertexArray(null); gl.activeTexture(gl.TEXTURE0);
  }
  advanceToTime(t, chunk) { this.advanceTo(Math.round(t / this.dt), chunk); }
  stateTex() { return this.S[this.cur]; } flipTex() { return this.F[this.cur]; }
  read(which = 'S') { const gl = this.gl; gl.bindFramebuffer(gl.FRAMEBUFFER, this.fb[this.cur]); gl.readBuffer(which === 'S' ? gl.COLOR_ATTACHMENT0 : gl.COLOR_ATTACHMENT1);
    const out = new Float32Array(this.w * this.h * 4); gl.readPixels(0, 0, this.w, this.h, gl.RGBA, gl.FLOAT, out); gl.readBuffer(gl.COLOR_ATTACHMENT0); gl.bindFramebuffer(gl.FRAMEBUFFER, null); return out; }
}
