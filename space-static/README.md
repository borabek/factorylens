---
title: FactoryLens
emoji: 🏭
colorFrom: gray
colorTo: blue
sdk: static
app_file: index.html
license: cc-by-nc-sa-4.0
short_description: RGB + 3D anomaly detection that notices a broken sensor
---

# FactoryLens demo (precomputed)

Anomaly detection on industrial parts with an RGB camera and a 3D sensor together. The fusion of the two notices, without labels, when one sensor starts failing, and stops trusting it.

This page shows precomputed results, made with the same code as the live demo in the repository (`factorylens demo`):

- **One part.** Four parts per product (one good, three defect types), clean or with one sensor broken in nine ways. For each, both sensors' anomaly maps, the fused map, the decision and how much each sensor counted.
- **A sensor fails.** Each product's test parts come down a simulated line and one sensor breaks halfway. You can follow the sensor weights and compare the alarms with plain late fusion.

Code, method and all results: [github.com/borabek/factorylens](https://github.com/borabek/factorylens) (commit COMMIT).

## Data and licence

All images, and the models that produced the maps (fitted on normal parts only), come from **MVTec 3D-AD**:

> Paul Bergmann, Xin Jin, David Sattlegger, Carsten Steger. *The MVTec 3D-AD Dataset for Unsupervised 3D Anomaly Detection and Localization.* VISAPP 2022.

MVTec 3D-AD is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). The images here were downscaled to 224 px, with the background plane removed from the 3D data, and the anomaly maps are derived from them. This Space and everything in it is shared under the same licence, **for non-commercial use only**. The FactoryLens code is MIT-licensed; this Space is not.
