# Energy Abundant System Challenge. 



For more than a century, industrial civilisation has been built around a simple assumption: energy is scarce. Factories, refineries, chemical plants and mines therefore became very good at conserving energy and operating continuously under predictable conditions.



We see a near future where this assumption changes shape. True energy abundance becomes a reality, but only at a particular time and place. Solar power has become extraordinarily cheap to deploy. As renewable generation continues to fall in cost, the key question has become: how do we redesign industrial systems around energy that is cheap, clean and abundant, while still intermittent? When computation became cheap, we stopped treating it as a scarce resource and created entirely new kinds of software. Cheap renewable energy should produce an equivalent transformation in industrial systems. The Commission II sponsor, Rivan, is developing a solar-powered synthetic-fuel system that converts water and carbon dioxide captured from the air into methane. The system must coordinate several processes with different power requirements, thermal dynamics, duty cycles and operating constraints:



1. Direct-air capture through calcium looping
2. Carbonation of calcium oxide
3. Calcination of calcium carbonate in a kiln
4. Hydrogen production through electrolysis
5. Methanation through a Sabatier reactor



Unlike a conventional refinery, such a plant cannot assume a constant supply of energy. Its behaviour depends on solar generation, temperature, humidity, weather forecasts, equipment condition and the state of every process within it. A plant deployed in a remote solar field will also have minimal personnel permanently on site. So remote solar plants are only economical if they can run themselves.



## The Challenge

Design, model or build part of the autonomous industrial plant of the future. We are looking for hardware and/or software work that helps a remote, solar-powered plant operate productively with intermittent power, uncertain weather, equipment faults and limited human supervision. The synthetic fuel plant provides a concrete reference system, but submissions do not need to reproduce Rivan’s architecture exactly. We welcome projects addressing other energy-intensive industrial processes where abundant but variable renewable power changes how the system should be designed.



## Suggested Areas of Exploration
### Reference Digital-Twin Challenge

For entrants who prefer a well-defined software and controls problem, one possible submission is an autonomous digital twin of the Rivan plant.



Build an application that accepts:



* A location in Europe
* Solar-array and battery capacity
* Simplified operating curves for each plant subsystem
* A 10-day weather forecast



The application should simulate plant operation, choose a production schedule and report:



Synthetic-methane production and plant utilisation.



* Battery state over time
* Energy lost through curtailment
* The limiting subsystem
* Responses to at least one injected fault



Compare the autonomous strategy against at least one simple baseline, such as operating every subsystem whenever power is available.



This is intended as a starting point for entrants who want one, not as a restriction on the wider commission.



### Plant Autonomy and Control

Build a system that determines how a plant or subsystem should operate.



Questions might include:



1. How should the plant schedule subsystems and allocate available electricity between immediate use, storage and future demand?
2. How should operation adapt when forecasts are wrong or environmental conditions change?
3. Can the system recognise, isolate and recover from faults without human intervention?



Possible approaches include:



* Model-predictive control and optimisation
* Rule-based or PLC/embedded control
* Reinforcement learning
* Distributed control architectures



### Siting and Production Forecasting

Build a tool that determines where plants should be deployed and what they could produce.



Inputs might include:



* GPS coordinates
* Historical or forecast weather
* Solar-resource data
* Temperature and humidity
* Land and grid constraints
* Plant capacity
* Storage capacity
* Capital and operating costs



Outputs might include:



* Methane, hydrogen or captured-CO₂ production
* Plant utilisation
* Energy curtailed or stored
* Production cost
* Expected time to profitability
* Limiting equipment
* Recommended plant locations



## What We Would Like to See

Strong submissions will demonstrate some combination of the following:



1. A real technical artefact: Build something that can be inspected, run or tested.
2. Clear assumptions: Distinguishing between supplied data, measured data, assumptions, and invented values.
3. Realistic operating constraints: Considering uncertainty, degradation, and failures.



## Submission Requirements

Please provide:



1. A GitHub repository (publicly shared) or design package that can include the code, CAD, schematics, BoM, or other technical materials needed to understand and reproduce the project.
2. A 1-5 minute video, slide deck, or working application showing the project. For simulation and controls focused submissions, show the system responding dynamically rather than only presenting final results.
3. A 2-page maximum writeup covering your motivation, chosen problem, architecture, and the assumptions you made. Explain what worked, what failed, and what you would do next to develop the project.



## Judging

We care about ideas and execution. Hacky but insightful prototypes are welcome, alongside highly developed systems.



Submissions will be assessed by expert judges on:



* Technical excellence: Is the project well designed and competently executed?
* Novelty: Does it reveal a new architecture, mechanism, control strategy or deployment model?
* Feasibility: Could the approach plausibly be developed into a real system?
* Adherence to the brief: Does the submission address a real bottleneck in making remote, renewable-powered industrial plants practical?

