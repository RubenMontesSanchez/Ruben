"use client";

import { Suspense, useRef } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, useGLTF, Environment, Center } from "@react-three/drei";
import * as THREE from "three";

function Model({ url }: { url: string }) {
  const { scene } = useGLTF(url);
  const ref = useRef<THREE.Group>(null);

  useFrame((_, delta) => {
    if (ref.current) ref.current.rotation.y += delta * 0.4;
  });

  return (
    <Center>
      <primitive ref={ref} object={scene} />
    </Center>
  );
}

function Placeholder() {
  return (
    <mesh>
      <boxGeometry args={[1, 1, 1]} />
      <meshStandardMaterial color="#4f6ef7" wireframe />
    </mesh>
  );
}

interface Props {
  url: string | null;
}

export default function ModelViewer({ url }: Props) {
  return (
    <div className="w-full h-full rounded-2xl overflow-hidden glass">
      <Canvas
        camera={{ position: [0, 1, 3], fov: 50 }}
        style={{ background: "transparent" }}
      >
        <ambientLight intensity={0.6} />
        <directionalLight position={[5, 10, 5]} intensity={1.2} castShadow />
        <Environment preset="city" />
        <Suspense fallback={<Placeholder />}>
          {url ? <Model url={url} /> : <Placeholder />}
        </Suspense>
        <OrbitControls
          enablePan={false}
          minDistance={1}
          maxDistance={8}
          autoRotate={!url}
          autoRotateSpeed={1}
        />
      </Canvas>
      {!url && (
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <p className="text-slate-500 text-sm">El modelo aparecerá aquí</p>
        </div>
      )}
    </div>
  );
}
