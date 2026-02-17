"use client";

import React from 'react';
import { Card } from "@/components/ui/Card";
import { Math as MathBlock } from "@/components/mathematics/Math";

export function MatrixVis() {
    return (
        <Card variant="glass" className="p-8 my-8 bg-zinc-900/50">
            <div className="flex flex-col gap-8">
                <div className="text-center">
                    <h4 className="text-base font-semibold text-zinc-200 mb-2">Forward Pass in Matrix Notation</h4>
                    <p className="text-sm text-zinc-400">
                        Each layer k computes pre-activations by multiplying the previous layer's output hₖ₋₁ by a weight matrix Ωₖ and adding a bias vector βₖ.
                    </p>
                </div>

                <div className="flex flex-wrap justify-center items-center gap-8 overflow-x-auto py-4">

                    {/* Step 1: Linear Transform */}
                    <div className="flex flex-col items-center gap-3 bg-zinc-900/80 p-4 rounded-xl border border-zinc-800">
                        <span className="text-xs uppercase tracking-widest text-blue-400 font-bold">Linear Step</span>
                        <MathBlock latex="\mathbf{z}_k = \boldsymbol{\beta}_k + \boldsymbol{\Omega}_k \mathbf{h}_{k-1}" />
                        <div className="grid grid-cols-3 gap-1 text-[10px] text-zinc-500 font-mono mt-2">
                            <div className="text-center">Bias<br />(Dₖ × 1)</div>
                            <div className="text-center">Weights<br />(Dₖ × Dₖ₋₁)</div>
                            <div className="text-center">Input<br />(Dₖ₋₁ × 1)</div>
                        </div>
                    </div>

                    <div className="text-zinc-600">→</div>

                    {/* Step 2: Activation */}
                    <div className="flex flex-col items-center gap-3 bg-zinc-900/80 p-4 rounded-xl border border-zinc-800">
                        <span className="text-xs uppercase tracking-widest text-green-400 font-bold">Activation</span>
                        <MathBlock latex="\mathbf{h}_k = \text{ReLU}[\mathbf{z}_k]" />
                        <div className="text-[10px] text-zinc-500 font-mono mt-2 text-center">
                            Element-wise<br />Non-linearity
                        </div>
                    </div>

                </div>

                <div className="bg-zinc-950/50 p-4 rounded-lg border border-white/5">
                    <div className="flex gap-4 font-mono text-xs text-zinc-400 justify-center">
                        <div className="flex items-center gap-2">
                            <span className="w-2 h-2 rounded-full bg-blue-500"></span>
                            <span>Ω: Weights (Slopes)</span>
                        </div>
                        <div className="flex items-center gap-2">
                            <span className="w-2 h-2 rounded-full bg-purple-500"></span>
                            <span>β: Biases (Offsets)</span>
                        </div>
                        <div className="flex items-center gap-2">
                            <span className="w-2 h-2 rounded-full bg-green-500"></span>
                            <span>h: Activations</span>
                        </div>
                    </div>
                </div>
            </div>
        </Card>
    );
}
