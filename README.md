# Reconfigurable Intelligent Surfaces (RIS) for Lunar Communications

## Overview

NASA has identified candidate Artemis III landing regions near the lunar south pole [[nasa.gov]](https://www.nasa.gov/news-release/nasa-identifies-candidate-regions-for-landing-next-americans-on-moon/). The steep terrain gradients characteristic of the south pole create severe Non-Line-of-Sight (NLOS) attenuation zones, resulting in significant signal degradation for surface assets like Lunar Terrain Vehicles (LTVs).

To mitigate this challenge, it is proposed to deploy a Reconfigurable Intelligent Surface (RIS) on a 100-meter mast situated on proximate elevation peaks (modeled after Honeybee Robotics' LUNARSABER [[space.com]](https://www.space.com/researchers-want-to-build-streetlights-on-the-moon)). This architecture reflects the signal to illuminate NLOS valleys and maintain continuous connectivity.

This proposed architecture is evaluated using the **Malapert Massif** as an initial test case. Utilizing the **NVIDIA Sionna-RT** raytracing engine [[nvidia.com]](https://developer.nvidia.com/sionna), multipath propagation is computed over a high-resolution 3D digital elevation model (DEM) of the Massif. This calculates precise received power (dBm) across a receiver grid for both the standalone Starship Human Landing System (HLS) base station and the RIS-assisted topology. Pending orbital DEM data availability, this deterministic physical layer simulation pipeline will be scaled to model all Artemis III candidate regions. Results are exported to an interactive 3D spatial dashboard for coverage analysis.

---

## Environment
- **Terrain Import**: Ingest the 3D wavefront mesh of the Malapert Massif (`nasa_malapert_dem.obj`). *Justification: NASA's Lunar Orbiter Laser Altimeter (LOLA) provides the highest resolution topographical data for the lunar south pole (5 m/pixel) as archived by the [Planetary Geosciences Data Archive (PGDA)](https://pgda.gsfc.nasa.gov/products/78).*
- **Receiver Grid Generation**: Synthesize an 80x80 spatial grid of LTV receivers anchored to the terrain surface, plus a 1.5-meter vertical offset. *Justification: 1.5 meters approximates the standard communication mast height of lunar rovers and astronaut EVA suits ([Apollo Lunar Roving Vehicle Specs](https://en.wikipedia.org/wiki/Lunar_Roving_Vehicle)).*
- **Dielectric Properties**: Initialize the RF material properties for Lunar Regolith. The engine uses a relative permittivity of ε_r ≈ 3.1 to model scattering and reflection. *Justification: As established in the [Lunar Sourcebook (Carrier, Olhoeft, & Mendell, 1991)](https://www.lpi.usra.edu/publications/books/lunar_sourcebook/), "the relative permittivity of lunar soil is a function of its bulk density" according to the empirical relationship ε_r = 1.93^ρ. Assuming a standard regolith density of ρ = 1.72 g/cm³, the relative permittivity mathematically evaluates to 3.1.*

## Transmitter Topology
- **Base Station (BS)**: Anchor the Starship HLS Base Station on the Earth-facing slope at an elevation offset of 50 meters, transmitting at 43 dBm (20 Watts) using a 3GPP `tr38901` 8x8 planar array. *Justification: 50m derives from official SpaceX Starship physical specifications ([SpaceX Starship](https://en.wikipedia.org/wiki/SpaceX_Starship)). 43 dBm and the `tr38901` array are the global benchmarks for 5G macro-cells defined in [ETSI TR 138 901](https://www.etsi.org/deliver/etsi_tr/138900_138999/138901/19.04.00_60/tr_138901v190400p.pdf).*
- **RIS Relay Node**: Anchor the Reconfigurable Intelligent Surface at the massif apex on a 100-meter mast, with an active power budget of 36 dBm. *Justification: 100m assumes deployment of Honeybee Robotics' LUNARSABER concept. 36 dBm (~4 Watts) is the standard physical power benchmark utilized in recent academic literature for analyzing active RIS array performance ([Active RIS vs. Passive RIS: Which Will Prevail in 6G?](https://arxiv.org/abs/2103.15154)).*

## Propagation Modeling
For discrete frequency bands (3.5 GHz, 10 GHz, and 28 GHz), the deterministic raytracing engine computes the multipath channel impulse response across the terrain to evaluate the spatial power distribution. *Justification: These bands are selected based on 3GPP Non-Terrestrial Network specifications ([TR 38.811](https://portal.3gpp.org/desktopmodules/Specifications/SpecificationDetails.aspx?specificationId=3234)) and standard X-band deep space links.*
- **BS Direct Raycasting**: Compute the received signal power (dBm) for all 6,400 receiver nodes by synthesizing Line-of-Sight (LOS) and Non-Line-of-Sight (NLOS) multipath propagation from the Base Station.
- **RIS Relay Raycasting**: Compute the equivalent signal strength parameters propagated via the active RIS Relay node.
- **Coverage Synthesis**: The aggregated received power per node is determined via Selection Combining (SC), evaluating the maximum instantaneous Signal-to-Noise Ratio (SNR) between the direct BS and relayed RIS propagation branches. *Justification: This standard algorithm mathematically models [Selection Combining (SC) diversity](https://en.wikipedia.org/wiki/Diversity_combining#Selection_combining), where the receiver seamlessly switches to the best connection to combat spatial fading.*

## Data Export & Visualization
- **Data Serialization**: Export the spatial coordinates and computed dBm arrays to a local SQLite database (`lunar_telemetry.db`) and a compressed NumPy archive (`malapert_coverage_data.npz`).
- **Dashboard Generation**: Instantiate a Plotly-based 3D geospatial dashboard to map the analytical data:
  - Render the underlying terrain mesh.
  - Render 3D representations of the HLS and RIS infrastructure.
  - Plot the receiver grid as a heatmapped point cloud based on received power (dBm).
  - Implement dynamic UI toggles to visually compare the baseline HLS coverage against the RIS-assisted topology.
