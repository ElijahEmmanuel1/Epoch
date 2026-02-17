"use client";

import React, { useState, useEffect, useRef } from 'react';
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Layers, RefreshCw } from "lucide-react";
import { Math as MathBlock } from "@/components/mathematics/Math";

// Simple Recharts or Canvas for the graph. Since we need high performance/interactivity, canvas is better.
// But as per user request for "non-ASCII schemas", we will use SVG for crisp rendering.

interface HiddenUnit {
    id: number;
    w0: number; // Input weight (slope)
    b: number; // Bias (offset)
    w1: number; // Output weight
}

export function ShallowNetViz() {
    const [units, setUnits] = useState<HiddenUnit[]>([
        { id: 1, w0: 1.5, b: -0.5, w1: 1.0 },
        { id: 2, w0: -1.0, b: 0, w1: 0.5 },
        { id: 3, w0: 0.5, b: 0.5, w1: -1.0 },
    ]);
    const [biasOutput, setBiasOutput] = useState(0);
    const [activeUnit, setActiveUnit] = useState<number | null>(null);

    // SVG dimensions
    const width = 600;
    const height = 400;
    const padding = 40;

    // Coordinate mapping
    const xRange = [-2, 2];
    const yRange = [-2, 2];

    const mapX = (x: number) => padding + ((x - xRange[0]) / (xRange[1] - xRange[0])) * (width - 2 * padding);
    const mapY = (y: number) => height - padding - ((y - yRange[0]) / (yRange[1] - yRange[0])) * (height - 2 * padding);

    // Function: ReLU(w0 * x + b) * w1
    const computeUnitOutput = (x: number, unit: HiddenUnit) => {
        const preActivation = Math.max(0, unit.w0 * x + unit.b);
        const activation = Math.max(0, preActivation);
        return activation * unit.w1;
    };

    const computeTotalOutput = (x: number) => {
        let sum = biasOutput;
        units.forEach(u => sum += computeUnitOutput(x, u));
        return sum;
    };

    // Generate path data
    const generatePath = (fn: (x: number) => number) => {
        const points: [number, number][] = [];
        const steps = 100;
        for (let i = 0; i <= steps; i++) {
            const t = i / steps;
            const x = xRange[0] + t * (xRange[1] - xRange[0]);
            const y = fn(x);
            points.push([mapX(x), mapY(y)]);
        }
        return `M ${points.map(p => `${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(" L ")}`;
    };

    const updateUnit = (id: number, key: keyof HiddenUnit, value: number) => {
        setUnits(prev => prev.map(u => u.id === id ? { ...u, [key]: value } : u));
    };

    return (
        <Card variant="bordered" className="p-6 my-8 bg-zinc-900/50 border-zinc-800">
            <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-2">
                    <Layers className="h-5 w-5 text-blue-500" />
                    <h3 className="font-semibold text-zinc-100">Interactive Shallow Network</h3>
                </div>
                <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => {
                        setUnits([
                            { id: 1, w0: 1.5, b: -0.5, w1: 1.0 },
                            { id: 2, w0: -1.0, b: 0, w1: 0.5 },
                            { id: 3, w0: 0.5, b: 0.5, w1: -1.0 },
                        ]);
                        setBiasOutput(0);
                    }}
                    className="text-xs text-zinc-500"
                >
                    <RefreshCw className="h-3 w-3 mr-1" />
                    Reset
                </Button>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">

                {/* Controls */}
                <div className="space-y-6 lg:max-h-[400px] lg:overflow-y-auto pr-2 scrollbar-thin">
                    <div className="p-4 rounded-lg bg-zinc-900 border border-zinc-800">
                        <label className="text-xs font-medium text-zinc-400 mb-2 block">Output Bias (β₀)</label>
                        <div className="flex items-center gap-3">
                            <input
                                type="range"
                                min="-2" max="2" step="0.1"
                                value={biasOutput}
                                onChange={(e) => setBiasOutput(parseFloat(e.target.value))}
                                className="flex-1 accent-indigo-500 h-1.5 bg-zinc-700 rounded-lg appearance-none cursor-pointer"
                            />
                            <span className="text-xs font-mono w-8 text-right text-zinc-300">{biasOutput.toFixed(1)}</span>
                        </div>
                    </div>

                    {units.map((unit, idx) => (
                        <div
                            key={unit.id}
                            className={`p-4 rounded-lg border transition-all duration-200 ${activeUnit === unit.id ? 'bg-blue-900/20 border-blue-500/50' : 'bg-zinc-900 border-zinc-800 hover:border-zinc-700'}`}
                            onMouseEnter={() => setActiveUnit(unit.id)}
                            onMouseLeave={() => setActiveUnit(null)}
                        >
                            <div className="flex items-center justify-between mb-3 border-b border-zinc-800 pb-2">
                                <span className="text-sm font-semibold text-zinc-200">Hidden Unit {idx + 1}</span>
                                <div className={`h-2 w-2 rounded-full ${['bg-red-500', 'bg-green-500', 'bg-yellow-500'][idx % 3]}`} />
                            </div>

                            <div className="space-y-3">
                                {/* Slope w0 */}
                                <div>
                                    <div className="flex justify-between text-[10px] text-zinc-500 mb-1">
                                        <span>Input Slope (ω)</span>
                                        <span className="font-mono text-zinc-300">{unit.w0.toFixed(1)}</span>
                                    </div>
                                    <input
                                        type="range"
                                        min="-3" max="3" step="0.1"
                                        value={unit.w0}
                                        onChange={(e) => updateUnit(unit.id, 'w0', parseFloat(e.target.value))}
                                        className="w-full accent-zinc-500 h-1.5 bg-zinc-800 rounded-lg appearance-none cursor-pointer"
                                    />
                                </div>

                                {/* Bias b */}
                                <div>
                                    <div className="flex justify-between text-[10px] text-zinc-500 mb-1">
                                        <span>Input Bias (β)</span>
                                        <span className="font-mono text-zinc-300">{unit.b.toFixed(1)}</span>
                                    </div>
                                    <input
                                        type="range"
                                        min="-2" max="2" step="0.1"
                                        value={unit.b}
                                        onChange={(e) => updateUnit(unit.id, 'b', parseFloat(e.target.value))}
                                        className="w-full accent-zinc-500 h-1.5 bg-zinc-800 rounded-lg appearance-none cursor-pointer"
                                    />
                                </div>

                                {/* Output Weight w1 */}
                                <div>
                                    <div className="flex justify-between text-[10px] text-zinc-500 mb-1">
                                        <span>Output Weight (ν)</span>
                                        <span className="font-mono text-zinc-300">{unit.w1.toFixed(1)}</span>
                                    </div>
                                    <input
                                        type="range"
                                        min="-2" max="2" step="0.1"
                                        value={unit.w1}
                                        onChange={(e) => updateUnit(unit.id, 'w1', parseFloat(e.target.value))}
                                        className="w-full accent-zinc-500 h-1.5 bg-zinc-800 rounded-lg appearance-none cursor-pointer"
                                    />
                                </div>
                            </div>
                        </div>
                    ))}
                </div>

                {/* Visualization */}
                <div className="lg:col-span-2 bg-black/40 rounded-xl border border-zinc-800/50 backdrop-blur-sm relative overflow-hidden flex items-center justify-center p-4">
                    <svg width="100%" height="100%" viewBox={`0 0 ${width} ${height}`} className="overflow-visible">
                        {/* Axes */}
                        <line x1={padding} y1={height / 2} x2={width - padding} y2={height / 2} stroke="#3f3f46" strokeWidth="1" />
                        <line x1={width / 2} y1={padding} x2={width / 2} y2={height - padding} stroke="#3f3f46" strokeWidth="1" />

                        {/* Individual Unit Paths (Faint) */}
                        {units.map((unit, idx) => (
                            <path
                                key={unit.id}
                                d={generatePath((x) => computeUnitOutput(x, unit))}
                                fill="none"
                                stroke={['#ef4444', '#22c55e', '#eab308'][idx % 3]}
                                strokeWidth="2"
                                strokeOpacity={activeUnit === unit.id ? 0.8 : 0.2}
                                strokeDasharray="4 4"
                            />
                        ))}

                        {/* Total Output Path (Strong) */}
                        <path
                            d={generatePath(computeTotalOutput)}
                            fill="none"
                            stroke="#6366f1"
                            strokeWidth="3"
                            className="drop-shadow-[0_0_8px_rgba(99,102,241,0.5)]"
                        />

                        {/* Labels */}
                        <text x={width - padding} y={height / 2 + 20} fill="#71717a" fontSize="12" textAnchor="end">x</text>
                        <text x={width / 2 + 10} y={padding} fill="#71717a" fontSize="12">y</text>
                    </svg>

                    <div className="absolute bottom-4 right-4 bg-black/60 backdrop-blur px-3 py-2 rounded-lg border border-white/10 text-xs font-mono text-zinc-400">
                        <div>Total Output</div>
                        <div className="text-indigo-400 font-bold">f(x; φ)</div>
                    </div>
                </div>
            </div>

            <div className="mt-6 pt-6 border-t border-zinc-800">
                <MathBlock latex="f(x) = \beta_0 + \sum \nu_j \cdot \text{ReLU}(\beta_j + \omega_j x)" className="text-zinc-500 text-sm" />
            </div>
        </Card>
    );
}
