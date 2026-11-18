import { useEffect, useRef, useState, Component, type ReactNode } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { Vector3 } from "three";
import VehicleModel, { type ModelProps } from "./VehicleModel";

export interface CameraCommand { kind: "reset" | "left" | "right" | "in" | "out" | "inspect"; serial: number; }
interface SceneProps extends ModelProps { focusCategory: ModelProps["selected"]; command: CameraCommand; reducedMotion: boolean; rotating: boolean; visible: boolean; onInteract: () => void; }

function Camera({ focusCategory: selected, command, reducedMotion, rotating, visible, onInteract }: SceneProps) {
  const { camera, gl, invalidate } = useThree();
  const control = useRef<OrbitControls | null>(null);
  const tween = useRef<{ from: Vector3; to: Vector3; targetFrom: Vector3; targetTo: Vector3; start: number } | null>(null);
  useEffect(() => {
    const c = new OrbitControls(camera, gl.domElement);
    c.enablePan = false; c.enableDamping = false;
    c.minDistance = 4.1; c.maxDistance = 12;
    c.minPolarAngle = .55; c.maxPolarAngle = Math.PI / 2 - .04;
    c.target.set(0, 1, 0); c.update();
    const start = () => { tween.current = null; onInteract(); };
    const change = () => invalidate();
    c.addEventListener("change", change); c.addEventListener("start", start);
    control.current = c;
    return () => { c.removeEventListener("change", change); c.removeEventListener("start", start); c.dispose(); control.current = null; };
  }, [camera, gl, invalidate, onInteract]);
  useEffect(() => {
    const c = control.current; if (!c) return;
    let to = new Vector3(-6.4, 3.7, 6.4), target = new Vector3(0, 1, 0);
    if (selected === "engine") { to.set(-4.5, 3.5, 4.3); target.set(-1.2, 1.1, 0); }
    if (selected === "brakes" || selected === "steering") { to.set(-4.2, 2.4, 4.8); target.set(-.9, .8, 0); }
    if (selected === "restraints") { to.set(-3.8, 4, 4.5); target.set(0, 1.5, 0); }
    if (command.kind === "inspect") { to.set(-5.5, 2.9, 6.2); target.set(0, 1, 0); }
    if (["left", "right", "in", "out"].includes(command.kind)) {
      target = c.target.clone(); to = camera.position.clone().sub(target);
      if (command.kind === "left" || command.kind === "right") to.applyAxisAngle(new Vector3(0, 1, 0), command.kind === "left" ? .35 : -.35);
      else to.setLength(Math.max(c.minDistance, Math.min(c.maxDistance, to.length() * (command.kind === "in" ? .82 : 1.22))));
      to.add(target);
    }
    if (reducedMotion || !visible) { camera.position.copy(to); c.target.copy(target); c.update(); }
    else tween.current = { from: camera.position.clone(), to, targetFrom: c.target.clone(), targetTo: target, start: performance.now() };
    invalidate();
  }, [selected, command, reducedMotion, camera, invalidate]);
  useEffect(() => { if (visible) invalidate(); }, [visible, rotating, invalidate]);
  useFrame(() => {
    const c = control.current; if (!c || !visible) return;
    if (tween.current) {
      const t = tween.current, elapsed = Math.min(1, (performance.now() - t.start) / 1100), ease = 1 - (1 - elapsed) ** 3;
      camera.position.lerpVectors(t.from, t.to, ease); c.target.lerpVectors(t.targetFrom, t.targetTo, ease); c.update();
      if (elapsed === 1) tween.current = null; else invalidate();
    } else if (rotating && !reducedMotion) {
      camera.position.sub(c.target).applyAxisAngle(new Vector3(0, 1, 0), .0015).add(c.target); c.update(); invalidate();
    }
  });
  return null;
}

class GraphicsBoundary extends Component<{ children: ReactNode; fallback: ReactNode }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() { return { failed: true }; }
  render() { return this.state.failed ? this.props.fallback : this.props.children; }
}

export default function VehicleScene(props: SceneProps) {
  const [lost, setLost] = useState(false);
  const fallback = <div className="vx-graphics-fallback"><img src="/images/vehicle-studio.svg" alt="Static generic vehicle illustration" /><p role="status">3D graphics are unavailable. The system list and evidence below remain available.</p></div>;
  return <GraphicsBoundary fallback={fallback}>{lost ? fallback : <Canvas
    aria-hidden="true"
    frameloop="demand" dpr={[1, 1.5]} camera={{ position: [-7.8, 4.3, 7.8], fov: 39 }}
    gl={{ antialias: true, alpha: false, powerPreference: "low-power" }} fallback={fallback}
    onCreated={({ gl }) => {
      const canvas = gl.domElement;
      const lose = (e: Event) => { e.preventDefault(); setLost(true); };
      canvas.addEventListener("webglcontextlost", lose, { once: true });
    }}
  >
    <color attach="background" args={["#0c1420"]} />
    <ambientLight intensity={.7} />
    <hemisphereLight args={["#d9edff", "#1d2839", 1.7]} />
    <directionalLight position={[-3, 7, 5]} intensity={3.2} color="#d8ecff" />
    <directionalLight position={[3, 5, -5]} intensity={4.5} color="#a4cce5" />
    <directionalLight position={[-5, 2, -3]} intensity={1.3} color="#ffffff" />
    <VehicleModel {...props} />
    <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, .035, 0]}><circleGeometry args={[4.2, 80]} /><meshStandardMaterial color="#142235" roughness={.8} metalness={.25} /></mesh>
    <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, .04, 0]}><ringGeometry args={[3.9, 3.915, 80]} /><meshBasicMaterial color="#31516c" /></mesh>
    <Camera {...props} />
  </Canvas>}</GraphicsBoundary>;
}
