import React, { useState } from 'react';
import './TensorRTVisualization.css';

function TensorRTVisualization() {
  const [currentStep, setCurrentStep] = useState(0);

  const steps = [
    {
      title: "Step 1: Model Tracing",
      code: "traced_model_fp16 = torch.jit.trace(model, torch.randn(1, 1, 28, 28).cuda())",
      description: "Converting PyTorch model to TorchScript format using trace",
      details: "Creates a traced version of the model with sample input shape (1, 1, 28, 28) - batch size 1, 1 channel, 28x28 image"
    },
    {
      title: "Step 2: TensorRT Compilation",
      code: "trt_model_fp16 = torch_tensorrt.compile(...)",
      description: "Compiling traced model to TensorRT with FP16 precision",
      details: "TensorRT optimizes the model for NVIDIA GPUs using graph optimization and kernel fusion"
    },
    {
      title: "Step 3: Input Configuration",
      code: "inputs=[torch_tensorrt.Input(shape=[1, 1, 28, 28])]",
      description: "Defining input tensor shape",
      details: "Shape: [Batch, Channels, Height, Width] = [1, 1, 28, 28] for MNIST grayscale images"
    },
    {
      title: "Step 4: FP16 Precision",
      code: "enabled_precisions={torch.half}",
      description: "Using 16-bit floating point precision",
      details: "FP16 uses half the memory of FP32 and provides ~2x speedup with minimal accuracy loss"
    },
    {
      title: "Step 5: Workspace Allocation",
      code: "workspace_size=1 << 30",
      description: "Allocating 1GB workspace memory",
      details: "1 << 30 = 2^30 bytes = 1GB for TensorRT optimization algorithms"
    },
    {
      title: "Step 6: Model Saving",
      code: "torch.jit.save(trt_model_fp16, '../models/mnist_trt_fp16.ts')",
      description: "Saving compiled TensorRT model",
      details: "Saves optimized model to disk for later inference without recompilation"
    }
  ];

  const quantizationMethods = [
    {
      method: "FP32 (Full Precision)",
      bits: "32-bit",
      memory: "100%",
      speed: "1x",
      accuracy: "100%",
      color: "#3b82f6"
    },
    {
      method: "FP16 (Half Precision)",
      bits: "16-bit",
      memory: "50%",
      speed: "2x",
      accuracy: "~99.9%",
      color: "#10b981"
    },
    {
      method: "INT8 (Integer)",
      bits: "8-bit",
      memory: "25%",
      speed: "4x",
      accuracy: "~98-99%",
      color: "#f59e0b"
    },
    {
      method: "INT4 (Ultra-compressed)",
      bits: "4-bit",
      memory: "12.5%",
      speed: "8x",
      accuracy: "~95-97%",
      color: "#ef4444"
    }
  ];

  return (
    <div className="container">
      <h1>TensorRT FP16 Compilation Visualizer</h1>
      
      {/* Code Flow Visualization */}
      <div className="section">
        <h2>Compilation Process</h2>
        <div className="steps-container">
          {steps.map((step, index) => (
            <div 
              key={index} 
              className={`step ${currentStep === index ? 'active' : ''}`}
              onClick={() => setCurrentStep(index)}
            >
              <div className="step-header">
                <span className="step-number">{index + 1}</span>
                <h3>{step.title}</h3>
              </div>
              <div className="code-block">{step.code}</div>
              <p className="description">{step.description}</p>
              {currentStep === index && (
                <div className="details">{step.details}</div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Quantization Comparison */}
      <div className="section">
        <h2>Quantization Methods Comparison</h2>
        <div className="quantization-grid">
          {quantizationMethods.map((method, index) => (
            <div key={index} className="quant-card" style={{borderColor: method.color}}>
              <h3 style={{color: method.color}}>{method.method}</h3>
              <div className="quant-details">
                <div className="detail-item">
                  <span className="label">Bits:</span>
                  <span className="value">{method.bits}</span>
                </div>
                <div className="detail-item">
                  <span className="label">Memory:</span>
                  <span className="value">{method.memory}</span>
                </div>
                <div className="detail-item">
                  <span className="label">Speed:</span>
                  <span className="value">{method.speed}</span>
                </div>
                <div className="detail-item">
                  <span className="label">Accuracy:</span>
                  <span className="value">{method.accuracy}</span>
                </div>
              </div>
              <div className="progress-bar">
                <div 
                  className="progress-fill" 
                  style={{
                    width: method.memory,
                    backgroundColor: method.color
                  }}
                ></div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Architecture Diagram */}
      <div className="section">
        <h2>TensorRT Optimization Pipeline</h2>
        <div className="pipeline">
          <div className="pipeline-step">
            <div className="box">PyTorch Model</div>
            <div className="arrow">→</div>
          </div>
          <div className="pipeline-step">
            <div className="box">TorchScript Trace</div>
            <div className="arrow">→</div>
          </div>
          <div className="pipeline-step">
            <div className="box">TensorRT Compile</div>
            <div className="arrow">→</div>
          </div>
          <div className="pipeline-step">
            <div className="box">FP16 Optimized</div>
            <div className="arrow">→</div>
          </div>
          <div className="pipeline-step">
            <div className="box">Saved Model</div>
          </div>
        </div>
      </div>

      {/* Key Benefits */}
      <div className="section">
        <h2>FP16 Quantization Benefits</h2>
        <div className="benefits-grid">
          <div className="benefit-card">
            <h3>⚡ 2x Faster Inference</h3>
            <p>Half-precision operations execute twice as fast on modern GPUs</p>
          </div>
          <div className="benefit-card">
            <h3>💾 50% Memory Savings</h3>
            <p>Reduces model size from 32-bit to 16-bit per parameter</p>
          </div>
          <div className="benefit-card">
            <h3>🎯 Minimal Accuracy Loss</h3>
            <p>Typically less than 0.1% accuracy degradation</p>
          </div>
          <div className="benefit-card">
            <h3>🚀 Better Throughput</h3>
            <p>Process more images per second with same hardware</p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default TensorRTVisualization;