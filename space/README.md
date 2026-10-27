---
title: FactoryLens
emoji: 🏭
colorFrom: gray
colorTo: blue
sdk: gradio
sdk_version: SDK_VERSION
python_version: "3.11"
app_file: app.py
license: cc-by-nc-sa-4.0
short_description: RGB + 3D anomaly detection that notices a broken sensor
---

# FactoryLens demo

Anomaly detection on industrial parts with an RGB camera and a 3D sensor together. The fusion of the two notices, without labels, when one sensor starts failing, and stops trusting it.

- **One part.** Pick a product and a test part, and optionally break one sensor. You see what each sensor's detector thinks, the fused decision and how much each sensor counted.
- **A sensor fails.** Parts come down a simulated line and one sensor breaks halfway. You can follow the sensor weights and compare the alarms with plain late fusion. On this free CPU Space a run uses 60 parts and takes about a minute.

Code, method and all results: [github.com/borabek/factorylens](https://github.com/borabek/factorylens) (commit COMMIT).

## Data and licence

The parts shown, and the models (fitted on normal parts only), come from **MVTec 3D-AD**:

> Paul Bergmann, Xin Jin, David Sattlegger, Carsten Steger. *The MVTec 3D-AD Dataset for Unsupervised 3D Anomaly Detection and Localization.* VISAPP 2022.

MVTec 3D-AD is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). The images here were downscaled to 224 px with the background plane removed, and the memory banks of the models hold features of the training images. This Space, its data and its models are shared under the same licence, **for non-commercial use only**. The FactoryLens code is MIT-licensed; this Space is not.
