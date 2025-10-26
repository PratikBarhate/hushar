## hushar [ हुशार ]

* A simple gRPC server for Machine Learning (ML) Model Inference in Rust.
* Name of the project, `हुशार`,  is the मराठी [ Marāṭhī ] translation of the word `intelligent`.
* `हुशार` uses [tract](https://github.com/sonos/tract) as the inference engine.


### High Level Details 

* The [service](schemas/protos/service.proto#L10) takes [InferenceRequest](schemas/protos/structs.proto#L7)
as an input payload and returns [InferenceResponse](schemas/protos/structs.proto#L12). 
* Server side metrics are collected an emitted by a [metrics side-car](hushar/src/io/side_car.rs#L11).
* [Feature logger side-car](hushar/src/io/side_car.rs#L62) writes the input features, modelId, requestId and model outputs to AWS S3



### Low Level Details 

#### Feature Processing

#### ML Model Inference

#### Feature Logging Side-Car

#### Metrics Emitting Side-Car

#### Experimentation



### Future Improvements



