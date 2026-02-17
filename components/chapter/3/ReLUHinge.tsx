"use client";

import React from 'react';
import { Card } from "@/components/ui/Card";

export function ReLUHinge() {
    const width = 400;
    const height = 250;
    const padding = 40;

    // Coordinate mapping
    const xRange = [-2, 2];
    const yRange = [-1, 3];

    const mapX = (x: number) => padding + ((x - xRange[0]) / (xRange[1] - xRange[0])) * (width - 2 * padding);
    const mapY = (y: number) => height - padding - ((y - yRange[0]) / (yRange[1] - yRange[0])) * (height - 2 * padding);

    const points = [
        { x: -2, y: 0 },
        { x: 0, y: 0 },
        { x: 2, y: 2 }
    ];

    const pathD = `M ${mapX(points[0].x)},${mapY(points[0].y)} L ${mapX(points[1].x)},${mapY(points[1].y)} L ${mapX(points[2].x)},${mapY(points[2].y)}`;

    return (
        <Card variant="glass" className="p-6 my-8 flex flex-col items-center bg-zinc-900/50">
            <h4 className="text-zinc-300 mb-4 font-mono text-sm">ReLU(z) = max(0, z)</h4>
            <svg width={width} height={height} className="overflow-visible">
                {/* Axes */}
                <line x1={padding} y1={mapY(0)} x2={width - padding} y2={mapY(0)} stroke="#3f3f46" strokeWidth="1" />
                <line x1={mapX(0)} y1={padding} x2={mapX(0)} y2={height - padding} stroke="#3f3f46" strokeWidth="1" />

                {/* ReLU Function */}
                <path d={pathD} fill="none" stroke="#22c55e" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />

                {/* Labels */}
                <text x={width - padding} y={mapY(0) + 20} textAnchor="end" fill="#71717a" fontSize="12">z</text>
                <text x={mapX(0) - 20} y={padding} textAnchor="end" fill="#71717a" fontSize="12">a</text>
            </svg>
            <p className="text-zinc-500 text-xs mt-4 text-center max-w-sm">
                La fonction ReLU "éteint" les valeurs négatives (z < 0) et laisse passer les valeurs positives linéairement.
            </p>
        </Card>
    );
}
