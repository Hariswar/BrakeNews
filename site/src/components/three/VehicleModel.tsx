import { useEffect, useMemo } from "react";
import { CanvasTexture, ExtrudeGeometry, Shape, SRGBColorSpace } from "three";
import type { ThreeEvent } from "@react-three/fiber";
import type { CategoryKey, ModelFamily } from "../../lib/visualization/types";

export interface ModelProps {
  family: ModelFamily;
  selected: CategoryKey | null;
  tones: Partial<Record<CategoryKey, number>>;
  onSelect: (key: CategoryKey) => void;
  decorative?: boolean;
}

function NumberMarker({ number, position, selected, onClick }: { number: number; position: [number, number, number]; selected: boolean; onClick: (e: ThreeEvent<MouseEvent>) => void }) {
  const texture = useMemo(() => {
    const canvas = document.createElement("canvas");
    canvas.width = canvas.height = 128;
    const c = canvas.getContext("2d")!;
    c.beginPath(); c.arc(64, 64, 53, 0, Math.PI * 2);
    c.fillStyle = selected ? "#d4f5ff" : "#102b43"; c.fill();
    c.strokeStyle = "#90d9f4"; c.lineWidth = 5; c.stroke();
    c.fillStyle = selected ? "#102b43" : "#ecfaff";
    c.font = "bold 58px sans-serif"; c.textAlign = "center"; c.textBaseline = "middle";
    c.fillText(String(number), 64, 68);
    const map = new CanvasTexture(canvas);
    map.colorSpace = SRGBColorSpace;
    return map;
  }, [number, selected]);
  useEffect(() => () => texture.dispose(), [texture]);
  return <sprite position={position} scale={[.38, .38, 1]} onClick={onClick} renderOrder={10}>
    <spriteMaterial map={texture} depthTest={false} transparent toneMapped={false} />
  </sprite>;
}

export default function VehicleModel({ family, selected, tones, onSelect, decorative = false }: ModelProps) {
  const suv = family === "suv", pickup = family === "pickup";
  const belt = suv || pickup ? 1.32 : 1.16;
  const roof = suv ? 2.15 : pickup ? 2.04 : 1.86;
  const rearRoof = pickup ? .15 : suv ? 1.65 : .8;
  const rearBase = pickup ? .55 : suv ? 2.05 : 1.55;
  const geometry = useMemo(() => {
    const s = new Shape();
    s.moveTo(-2.7, .62); s.lineTo(-2.7, belt - .16);
    s.quadraticCurveTo(-2.6, belt + .03, -2.24, belt + .04);
    s.lineTo(1.9, belt + .04); s.quadraticCurveTo(2.63, belt + .04, 2.67, belt - .15);
    s.lineTo(2.67, .62); s.lineTo(2.2, .62);
    s.bezierCurveTo(2.18, 1.36, 1.08, 1.36, 1.08, .62);
    s.lineTo(-1.08, .62); s.bezierCurveTo(-1.08, 1.36, -2.2, 1.36, -2.2, .62); s.closePath();
    return new ExtrudeGeometry(s, { depth: 1.76, bevelEnabled: true, bevelSize: .065, bevelThickness: .065, bevelSegments: 3, steps: 1, curveSegments: 16 });
  }, [belt]);
  const cabin = useMemo(() => {
    const s = new Shape();
    s.moveTo(-1.33, belt + .05); s.lineTo(-.65, roof - .1);
    s.quadraticCurveTo(-.54, roof, -.38, roof); s.lineTo(rearRoof, roof);
    s.lineTo(rearBase, belt + .06); s.closePath();
    return new ExtrudeGeometry(s, { depth: 1.57, bevelEnabled: true, bevelSize: .025, bevelThickness: .035, bevelSegments: 2, steps: 1 });
  }, [belt, roof, rearRoof, rearBase]);
  useEffect(() => () => geometry.dispose(), [geometry]);
  useEffect(() => () => cabin.dispose(), [cabin]);
  const click = (key: CategoryKey) => (e: ThreeEvent<MouseEvent>) => { e.stopPropagation(); if (!decorative) onSelect(key); };
  const paint = (key: CategoryKey, base = "#708b9f") => selected === key ? "#80d5ed" : (tones[key] ?? 0) > 0 ? `hsl(200, ${40 + (tones[key] ?? 0) * 35}%, ${40 + (tones[key] ?? 0) * 22}%)` : base;
  const emission = (key: CategoryKey) => selected === key ? "#17657d" : "#000000";
  const wheelX = [-1.64, 1.64];
  return <group name={`generic-${family}`}>
    <mesh geometry={geometry} position={[0, 0, -.88]} onClick={click("body")} castShadow>
      <meshPhysicalMaterial color={paint("body")} metalness={.72} roughness={.29} clearcoat={.8} emissive={emission("body")} />
    </mesh>
    {selected === "body" && <lineSegments position={[0, 0, -.88]}>
      <edgesGeometry args={[geometry, 35]} /><lineBasicMaterial color="#b5efff" />
    </lineSegments>}
    <mesh geometry={cabin} position={[0, 0, -.785]} onClick={click("restraints")} castShadow>
      <meshPhysicalMaterial color={paint("restraints", "#132f42")} metalness={.35} roughness={.14} clearcoat={1} emissive={emission("restraints")} />
    </mesh>
    <mesh position={[(rearRoof - .4) / 2, roof + .025, 0]} onClick={click("body")} castShadow>
      <boxGeometry args={[rearRoof + .48, .07, 1.61]} /><meshPhysicalMaterial color={paint("body")} metalness={.65} roughness={.3} />
    </mesh>
    <mesh position={[-1.9, belt + .08, 0]} onClick={click("engine")}>
      <boxGeometry args={[1.25, .045, 1.62]} /><meshPhysicalMaterial color={paint("engine")} metalness={.7} roughness={.28} emissive={emission("engine")} />
    </mesh>
    {selected === "engine" && <mesh position={[-1.9, belt + .105, 0]}><boxGeometry args={[1.27, .05, 1.65]} /><meshBasicMaterial color="#adf0ff" wireframe /></mesh>}
    {[1, -1].map((side) => <group key={side}>
      <mesh position={[pickup ? -.3 : .1, (roof + belt) / 2, side * .812]}><boxGeometry args={[.075, roof - belt, .06]} /><meshStandardMaterial color="#688293" metalness={.65} roughness={.28} /></mesh>
      <mesh position={[-.8, belt - .02, side * 1.01]}><boxGeometry args={[.27, .12, .18]} /><meshStandardMaterial color={paint("body")} metalness={.6} roughness={.3} /></mesh>
      {(pickup ? [-.25] : [-.6, .8]).map((x) => <mesh key={x} position={[x, belt - .16, side * .956]}><boxGeometry args={[.22, .04, .025]} /><meshStandardMaterial color="#c1d4df" metalness={.8} roughness={.25} /></mesh>)}
      <mesh position={[pickup ? .4 : .2, belt - .3, side * .96]}><boxGeometry args={[.014, .58, .012]} /><meshStandardMaterial color="#233f51" /></mesh>
      {wheelX.map((x) => <group key={x} position={[x, .59, side * 1.0]}>
        <mesh rotation={[Math.PI / 2, 0, 0]} castShadow onClick={click("steering")}>
          <cylinderGeometry args={[.52, .52, .3, 32]} /><meshStandardMaterial color={paint("steering", "#101821")} roughness={.82} emissive={emission("steering")} />
        </mesh>
        <mesh position={[0, 0, side * .165]} rotation={[Math.PI / 2, 0, 0]} onClick={click("brakes")}>
          <cylinderGeometry args={[.37, .37, .045, 32]} /><meshStandardMaterial color={paint("brakes", "#8fa7b9")} metalness={.88} roughness={.28} emissive={emission("brakes")} />
        </mesh>
        <mesh position={[0, 0, side * .193]} rotation={[Math.PI / 2, 0, 0]} onClick={click("brakes")}><cylinderGeometry args={[.29, .29, .01, 32]} /><meshStandardMaterial color="#22303c" metalness={.75} roughness={.45} /></mesh>
        {Array.from({ length: 5 }, (_, i) => {
          const a = i * Math.PI * 2 / 5;
          return <mesh key={i} position={[Math.sin(a) * .13, Math.cos(a) * .13, side * .21]} rotation={[0, 0, -a]} onClick={click("brakes")}><boxGeometry args={[.055, .31, .035]} /><meshStandardMaterial color={paint("brakes", "#bfd0da")} metalness={.85} roughness={.25} /></mesh>;
        })}
        <mesh position={[0, 0, side * .23]} rotation={[Math.PI / 2, 0, 0]}><cylinderGeometry args={[.09, .09, .03, 16]} /><meshStandardMaterial color="#71889b" metalness={.8} roughness={.3} /></mesh>
      </group>)}
      <mesh position={[-2.74, belt - .08, side * .62]}><boxGeometry args={[.035, .09, .5]} /><meshStandardMaterial color="#e1f6ff" emissive="#9fdfff" emissiveIntensity={1.2} /></mesh>
      <mesh position={[2.7, belt - .1, side * .7]}><boxGeometry args={[.03, .13, .34]} /><meshStandardMaterial color="#e2988c" emissive="#9a251e" emissiveIntensity={.6} /></mesh>
    </group>)}
    <mesh position={[-2.76, belt - .34, 0]}><boxGeometry args={[.025, .19, 1.22]} /><meshStandardMaterial color="#101923" roughness={.5} /></mesh>
    <mesh position={[-2.78, belt - .27, 0]}><boxGeometry args={[.03, .025, 1.24]} /><meshStandardMaterial color="#b3cedb" metalness={.9} roughness={.2} /></mesh>
    {pickup && <mesh position={[1.35, belt + .08, 0]}><boxGeometry args={[1.85, .04, 1.55]} /><meshStandardMaterial color="#162531" roughness={.9} /></mesh>}
    {!decorative && <>
      <NumberMarker number={1} position={[-1.92, belt + .3, 0]} selected={selected === "engine"} onClick={click("engine")} />
      <NumberMarker number={2} position={[-1.64, .63, 1.32]} selected={selected === "brakes"} onClick={click("brakes")} />
      <NumberMarker number={3} position={[-.1, roof + .3, 0]} selected={selected === "restraints"} onClick={click("restraints")} />
      <NumberMarker number={4} position={[-1.3, belt + .28, 1.06]} selected={selected === "steering"} onClick={click("steering")} />
      <NumberMarker number={5} position={[1.25, belt + .22, -.9]} selected={selected === "body"} onClick={click("body")} />
    </>}
  </group>;
}
