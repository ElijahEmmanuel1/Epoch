"use client";

import React from 'react';
import { Card } from "@/components/ui/Card";

export function DeepNetworkDiagram() {
    const width = 700;
    const height = 350;

    // Node positions
    const layers = [
        { title: "Input (x)", nodes: [0, 1], x: 100, color: "#71717a" },
        { title: "Hidden 1 (h₁)", nodes: [0, 1, 2, 3], x: 250, color: "#3b82f6" }, // Blue
        { title: "Hidden 2 (h₂)", nodes: [0, 1, 2], x: 400, color: "#8b5cf6" }, // Purple
        { title: "Hidden 3 (h₃)", nodes: [0, 1, 2, 3], x: 550, color: "#ec4899" }, // Pink
        { title: "Output (y)", nodes: [0], x: 700, color: "#ef4444" } // Red
    ];

    const nodeRadius = 12;

    const getY = (index: number, total: number) => {
        const spacing = 45;
        const offset = (height - (total - 1) * spacing) / 2;
        return offset + index * spacing;
    };

    return (
        <Card variant="glass" className="p-6 my-8 flex flex-col items-center bg-zinc-900/50">
            <h3 className="text-lg font-semibold mb-4 text-zinc-100">Deep Neural Network Architecture</h3>
            <svg width={800} height={height} className="overflow-visible">
                <defs>
                    <marker id="arrow-deep" viewBox="0 0 10 10" refX="22" refY="5"
                        markerWidth="4" markerHeight="4" orient="auto-start-reverse">
                        <path d="M 0 0 L 10 5 L 0 10 z" fill="#52525b" opacity="0.4" />
                    </marker>
                </defs>

                {/* Connections */}
                {layers.map((layer, lIndex) => {
                    if (lIndex === layers.length - 1) return null;
                    const nextLayer = layers[lIndex + 1];

                    return layer.nodes.map(i => (
                        nextLayer.nodes.map(j => (
                            <line key={`l${lIndex}-n${i}-${j}`}
                                x1={layer.x} y1={getY(i, layer.nodes.length)}
                                x2={nextLayer.x} y2={getY(j, nextLayer.nodes.length)}
                                stroke="#52525b" strokeWidth="1" opacity="0.15"
                                markerEnd="url(#arrow-deep)"
                            />
                        ))
                    ));
                })}

                {/* Nodes */}
                {layers.map((layer, lIndex) => (
                    <g key={`layer-${lIndex}`}>
                        {layer.nodes.map((n) => (
                            <g key={`node-${lIndex}-${n}`}>
                                <circle
                                    cx={layer.x}
                                    cy={getY(n, layer.nodes.length)}
                                    r={nodeRadius}
                                    fill="#18181b"
                                    stroke={layer.color}
                                    strokeWidth="2"
                                />
                                <text
                                    x={layer.x}
                                    y={getY(n, layer.nodes.length)}
                                    dy="4"
                                    fontSize="9"
                                    fill={layer.color}
                                    textAnchor="middle"
                                    fontWeight="bold"
                                >
                                    {lIndex === 0 ? `x${n + 1}` : lIndex === layers.length - 1 ? `y` : `h${n + 1}`}
                                </text>
                            </g>
                        ))}

                        {/* Layer Labels */}
                        <text x={layer.x} y={height - 20} textAnchor="middle" fill={layer.color} fontSize="11" fontWeight="bold">
                            {layer.title}
                        </text>
                    </g>
                ))}

            </svg>
        </Card>
    );
}
