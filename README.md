# Physics-Informed Unsupervised Deep Learning for Detection and Isolation of Unseen Battery Overcharge Faults

This repository contains the official code and pre-trained model for the above maniscript. This work presents a novel diagnostic framework that integrates first-principles thermodynamics with an unsupervised 1D-CNN autoencoder to detect and isolate slight overcharge faults in lithium-ion batteries.

![Framework-Graphic](ToC.jpg)


The key innovation is a two-stage, physics-informed approach that is trained only on healthy battery data, eliminating the need for hazardous and difficult-to-acquire fault data.

The diagnostic methodology is split into two stages:

1. Unsupervised Anomaly Detection: A 1D-CNN Autoencoder (defined in the notebook) is trained exclusively on the discharge temperature profiles of a healthy battery (V42). Any deviation from this learned "healthy" thermal signature results in a high reconstruction error (RE), flagging an anomaly.

2. Physics-Informed Fault Isolation: A novel Overcharge Detection (OCD) metric, derived from thermodynamic analysis (Section 4.3 in the paper), is used to interpret the shape of the reconstruction error. This metric specifically identifies the unique thermal signature of an overcharge fault, distinguishing it from other anomalies (like over-discharge or low-temperature operation).
