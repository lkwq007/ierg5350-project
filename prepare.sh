#!/bin/bash
targets=(
    curl_rainbow
    dreamer
    drq
    midterm
    plan2explore
    ppo
    sac
)

for item in "${targets[@]}"; do
    cd "$item"
    ln -s ../gym_tetris gym_tetris
    ln -s ../gym_tetris_simple gym_tetris_simple
    cd ..
done