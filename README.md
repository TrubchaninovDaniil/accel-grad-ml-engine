## **Accel-Grad ML Engine**

**Accel-Grad** is an implementation of a high-performance machine learning optimizer built from scratch on C++. Designed for low-latency execution without external dependencies, the engine implements a custom mathematical core (RawMath) to handle mathematical operations.

### Architectural Framework

The engine utilizes a sigmoid-based predictive model and incorporates four distinct mathematical frameworks:

* **Gradient Descent:** first-order optimization for weight updates

* **Newton's Method (second-order optimization):** utilizes the second derivative of the loss  function to provide more precise step sizes

* **Generalized Lagrangian Function:** integrates a Lagrangian multiplier to manage and enforce functional constraints.

* **Momentum Dynamics:** implements a velocity-based update rule to accelerate convergence.

The engine is powered by a custom RawMath class, providing mathematical operations, including **Taylor series expansion** used for calculation of the exponential function $e^x$; **sigmoid function** as model and basic mathematical operations.

### **Mathematical Formulation**

#### The model calculates the prediction $Y$ using the sigmoid function:
$$ Y = \frac{1}{1 + e^{-(w \cdot x)}}$$

#### The **loss function** is defined as the mean squared error:
$$ \text{Loss} = \frac{1}{2} \| Y - \hat{Y} \|_2^2$$

#### We utilize the **Lagrangian function** to handle constraints:
$$ L(Y, \hat{Y}, \lambda) = \frac{1}{2} \| Y - \hat{Y} \|_2^2 + \lambda \cdot g(Y, \hat{Y}) $$

#### The gradient of the Lagrangian with respect to weights ($w$) is:
$$ \nabla_w L(w) = \underbrace{\nabla_w f(w)}_{\text{error}} + \underbrace{\lambda \cdot g(Y, \hat{Y})'}_{\text{constraint}} $$

#### The **velocity update rule (Momentum)** is defined as:
$$ v_{new} = v_{old} + \eta \left( v_{old} + \frac{\nabla_w L(w)}{\|\nabla_w f(w)\|_2^2 + \epsilon} \right) $$

Where:
* $v$ is the velocity (default = 0)
* $\epsilon$ is a very small value (e.g., $10^{-6}$)
* $\eta$ is the learning rate (step size)
