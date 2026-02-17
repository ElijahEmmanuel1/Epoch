"use client";

import React from 'react';
import { Card } from "@/components/ui/Card";

export function NetworkDiagram() {
    const width = 600;
    const height = 300;
    const padding = 40;

    // Node positions
    const inputX = 100;
    const hiddenX = 300;
    const outputX = 500;

    const inputs = [0, 1, 2]; // 3 inputs
    const hiddens = [0, 1, 2, 3]; // 4 hidden units
    const outputs = [0]; // 1 output

    const nodeRadius = 15;

    const getY = (index: number, total: number) => {
        const spacing = 50;
        const offset = (height - (total - 1) * spacing) / 2;
        return offset + index * spacing;
    };

    return (
        <Card variant="glass" className="p-6 my-8 flex justify-center items-center bg-zinc-900/50">
            <svg width={width} height={height} className="overflow-visible">
                <defs>
                    <marker id="arrow" viewBox="0 0 10 10" refX="28" refY="5"
                        markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                        <path d="M 0 0 L 10 5 L 0 10 z" fill="#52525b" opacity="0.5" />
                    </marker>
                </defs>

                {/* Connections Input -> Hidden */}
                {inputs.map((i) =>
                    hiddens.map((h) => {
                        const y1 = getY(i, inputs.length);
                        const y2 = getY(h, hiddens.length);
                        return (
                            <line key={`i${i}-h${h}`}
                                x1={inputX} y1={y1}
                                x2={hiddenX} y2={y2}
                                stroke="#52525b" strokeWidth="1" opacity="0.3"
                                markerEnd="url(#arrow)"
                            />
                        );
                    })
                )}

                {/* Connections Hidden -> Output */}
                {hiddens.map((h) =>
                    outputs.map((o) => {
                        const y1 = getY(h, hiddens.length);
                        const y2 = getY(o, outputs.length);
                        return (
                            <line key={`h${h}-o${o}`}
                                x1={hiddenX} y1={y1}
                                x2={outputX} y2={y2}
                                stroke="#52525b" strokeWidth="1" opacity="0.3"
                                markerEnd="url(#arrow)"
                            />
                        );
                    })
                )}

                {/* Nodes */}
                {inputs.map((i) => (
                    <g key={`input-${i}`}>
                        <circle cx={inputX} cy={getY(i, inputs.length)} r={nodeRadius} fill="#18181b" stroke="#71717a" strokeWidth="2" />
                        <text x={inputX - 30} y={getY(i, inputs.length)} dy="5" fontSize="12" fill="#a1a1aa" textAnchor="end">x_{i}</text>
                    </g>
                ))}

                {hiddens.map((h) => (
                    <g key={`hidden-${h}`}>
                        <circle cx={hiddenX} cy={getY(h, hiddens.length)} r={nodeRadius} fill="#18181b" stroke="#3b82f6" strokeWidth="2" />
                        <text x={hiddenX} y={getY(h, hiddens.length)} dy="5" fontSize="10" fill="#bae6fd" textAnchor="middle">h_{h}</text>
                    </g>
                ))}

                {outputs.map((o) => (
                    <g key={`output-${o}`}>
                        <circle cx={outputX} cy={getY(o, outputs.length)} r={nodeRadius} fill="#18181b" stroke="#ef4444" strokeWidth="2" />
                        <text x={outputX + 30} y={getY(o, outputs.length)} dy="5" fontSize="12" fill="#a1a1aa" textAnchor="start">y</text>
                    </g>
                ))}

                {/* Layer Labels */}
                <text x={inputX} y={height - 10} textAnchor="middle" fill="#71717a" fontSize="12" fontWeight="bold">Input Layer</text>
                <text x={hiddenX} y={height - 10} textAnchor="middle" fill="#3b82f6" fontSize="12" fontWeight="bold">Hidden Layer</text>
                <text x={outputX} y={height - 10} textAnchor="middle" fill="#ef4444" fontSize="12" fontWeight="bold">Output Layer</text>
            </svg>
        </Card>
    );
}
