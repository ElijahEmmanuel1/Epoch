"use client";

import { motion, type Variants } from "framer-motion";
import React from "react";

const fadeUp: Variants = {
    hidden: { opacity: 0, y: 24 },
    visible: (i: number) => ({
        opacity: 1,
        y: 0,
        transition: { delay: i * 0.1, duration: 0.5, ease: [0.25, 0.4, 0.25, 1] },
    }),
};

const fadeIn: Variants = {
    hidden: { opacity: 0 },
    visible: (i: number) => ({
        opacity: 1,
        transition: { delay: i * 0.1, duration: 0.6 },
    }),
};

const scaleIn: Variants = {
    hidden: { opacity: 0, scale: 0.95 },
    visible: (i: number) => ({
        opacity: 1,
        scale: 1,
        transition: { delay: i * 0.1, duration: 0.4, ease: [0.25, 0.4, 0.25, 1] },
    }),
};

interface AnimatedProps {
    children: React.ReactNode;
    className?: string;
    delay?: number;
    variant?: "fadeUp" | "fadeIn" | "scaleIn";
}

const variants = { fadeUp, fadeIn, scaleIn };

export function Animated({ children, className, delay = 0, variant = "fadeUp" }: AnimatedProps) {
    return (
        <motion.div
            custom={delay}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-64px" }}
            variants={variants[variant]}
            className={className}
        >
            {children}
        </motion.div>
    );
}

export function AnimatedGroup({ children, className }: { children: React.ReactNode; className?: string }) {
    return (
        <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-64px" }}
            className={className}
        >
            {React.Children.map(children, (child, i) => (
                <motion.div custom={i} variants={fadeUp}>
                    {child}
                </motion.div>
            ))}
        </motion.div>
    );
}

export function FloatingGlow({ className }: { className?: string }) {
    return (
        <motion.div
            className={className}
            animate={{
                scale: [1, 1.1, 1],
                opacity: [0.3, 0.5, 0.3],
            }}
            transition={{
                duration: 8,
                repeat: Infinity,
                ease: "easeInOut",
            }}
        />
    );
}
