---
title: Daylight Studio
emoji: ☀️
colorFrom: gray
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
short_description: Bubble diagram to daylight (sDA) prediction, MLP/XGBoost/GNN
---

# Daylight Studio

Interactive bubble-diagram editor with live daylight prediction for residential units (Shanghai and Singapore datasets).
Draw or pick a boundary, place room bubbles, set the window-to-wall ratios, and the trained MLP / XGBoost / GNN
surrogates return the DA map, room-level sDA and layout-level sDA.

- **Source code, models and data:** https://github.com/microclimate-design-lab/daylight_studio
- **User guide:** [HANDOVER.md](https://github.com/microclimate-design-lab/daylight_studio/blob/main/HANDOVER.md)

The predictions are fast early-stage estimates from models trained on simulated layouts; they do not replace a
daylight simulation. See the guide for the valid input range and known limitations.

This Space is a deployment copy of the GitHub repository; please open issues and changes there.
