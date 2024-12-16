## RNN


```mermaid


graph LR
    x1((x₁)) -->|"$$W_{xh}$$"| h1
    x2((x₂)) -->|"$$W_{xh}$$"| h2
    x3((x₃)) -->|"$$W_{xh}$$"| h3
    
    h0((h₀)) -->|"$$W_{hh}$$"| h1((h₁))
    h1 -->|"$$W_{hh}$$"| h2((h₂))
    h2 -->|"$$W_{hh}$$"| h3((h₃))
    
    h1 -->|"$$W_{hy}$$"| y1((y₁))
    h2 -->|"$$W_{hy}$$"| y2((y₂))
    h3 -->|"$$W_{hy}$$"| y3((y₃))
    
```

$$h_t = \phi(W_{xh}x_t + W_{hh}h_{t-1} + b_h) \tag{1}$$
$$y_t = \psi(W_{hy}h_t + b_y) \tag{2}$$



**Loss function**: $$L = \sum_{t=1}^{T} L(y_t, \hat{y}_t) \tag{3}$$



**Backpropagation through time (BPTT)**

$$\frac{\partial L}{\partial W_{hy}} = \sum_{t=1}^{T} \frac{\partial L}{\partial y_t} \frac{\partial y_t}{\partial W_{hy}} \tag{4}$$

$$\frac{\partial y_t}{\partial W_{hy}} = \psi'(z_t)h_t^T \tag{5}$$

where: $$z_t = W_{hy}h_t + b_y \tag{6}$$

$$\frac{\partial L}{\partial W_{hh}} = \sum_{t=1}^{T} \frac{\partial L}{\partial h_t} \frac{\partial h_t}{\partial W_{hh}} \tag{7}$$

$$\frac{\partial h_t}{\partial W_{hh}} = \phi'(z_t)h_{t-1}^T \tag{8}$$

where: $$z_t = W_{xh}x_t + W_{hh}h_{t-1} + b_h \tag{9}$$



```mermaid
graph LR
    x1((x₁)) -->|"Input"| LSTM1[LSTM₁]
    h0((h₀)) --> LSTM1
    c0((c₀)) --> LSTM1
    LSTM1 --> h1((h₁))
    LSTM1 --> c1((c₁))
    LSTM1 --> y1((ŷ₁))

    x2((x₂)) -->|"Input"| LSTM2[LSTM₂]
    h1 --> LSTM2
    c1 --> LSTM2
    LSTM2 --> h2((h₂))
    LSTM2 --> c2((c₂))
    LSTM2 --> y2((ŷ₂))

    x3((x₃)) -->|"Input"| LSTM3[LSTM₃]
    h2 --> LSTM3
    c2 --> LSTM3
    LSTM3 --> h3((h₃))
    LSTM3 --> c3((c₃))
    LSTM3 --> y3((ŷ₃))

```

```mermaid
graph LR
    subgraph LSTM over seq_len steps
        x1((x₁)) --> LSTM1
        LSTM1 --> h1((h₁))
        LSTM1 --> c1((c₁))
        
        x2((x₂)) --> LSTM2
        h1 --> LSTM2
        c1 --> LSTM2
        LSTM2 --> h2((h₂))
        LSTM2 --> c2((c₂))
        
        ... 
        xn((xₙ)) --> LSTMn
        h(n-1) --> LSTMn
        c(n-1) --> LSTMn
        LSTMn --> hn((hₙ))
        LSTMn --> cn((cₙ))
    end

    hn --> Wy[Linear Layer Wy]
    Wy --> y_pred((Y_pred))
```

```mermaid
graph LR
    subgraph LSTM_over_Sequence
        x1((x₁)) --> LSTM1[LSTM₁]
        h0((h₀)) --> LSTM1
        c0((c₀)) --> LSTM1
        LSTM1 --> h1((h₁))
        LSTM1 --> c1((c₁))
        
        x2((x₂)) --> LSTM2[LSTM₂]
        h1 --> LSTM2
        c1 --> LSTM2
        LSTM2 --> h2((h₂))
        LSTM2 --> c2((c₂))
        
        x3((x₃)) --> LSTM3[LSTM₃]
        h2 --> LSTM3
        c2 --> LSTM3
        LSTM3 --> h3((h₃))
        LSTM3 --> c3((c₃))
    end

    h3 --> Wy[Linear Layer Wy]
    Wy --> y_pred((Y_pred))
```
```mermaid
graph TD
    %% Inputs to the LSTM cell
    subgraph Inputs
        xt["x<sub>t</sub>"]
        ht_prev["h<sub>t-1</sub>"]
        ct_prev["c<sub>t-1</sub>"]
    end

    %% LSTM Cell Operations
    subgraph LSTM_Cell["LSTM Cell"]
        %% Compute z = x_t * W + h_{t-1} * U + b
        z["z = x<sub>t</sub>⋅W + h<sub>t-1</sub>⋅U + b"]

        %% Split z into gates
        zf["z<sub>f</sub>"]
        zi["z<sub>i</sub>"]
        zg["z<sub>g</sub>"]
        zo["z<sub>o</sub>"]

        z -->|Split into 4 parts| zf
        z --> zi
        z --> zg
        z --> zo

        %% Activation Functions for Gates
        ft["f<sub>t</sub> = σ(z<sub>f</sub>)"] 
        it["i<sub>t</sub> = σ(z<sub>i</sub>)"]
        gt["g<sub>t</sub> = tanh(z<sub>g</sub>)"]
        ot["o<sub>t</sub> = σ(z<sub>o</sub>)"]

        zf --> ft
        zi --> it
        zg --> gt
        zo --> ot

        %% Cell State Update
        ft_ct_prev["f<sub>t</sub> ⋅ c<sub>t-1</sub>"]
        it_gt["i<sub>t</sub> ⋅ g<sub>t</sub>"]
        ct["c<sub>t</sub> = f<sub>t</sub>⋅c<sub>t-1</sub> + i<sub>t</sub>⋅g<sub>t</sub>"]

        ft --> ft_ct_prev
        it --> it_gt
        ft_ct_prev --> ct
        it_gt --> ct

        %% Hidden State Update
        tanh_ct["tanh(c<sub>t</sub>)"]
        ht["h<sub>t</sub> = o<sub>t</sub> ⋅ tanh(c<sub>t</sub>)"]

        ct --> tanh_ct
        ot --> ht
        tanh_ct --> ht
    end

    %% Connecting Inputs to LSTM Cell
    xt --> LSTM_Cell
    ht_prev --> LSTM_Cell
    ct_prev --> LSTM_Cell

    %% Outputs from LSTM Cell
    ht --> Outputs["Outputs"]
    ct --> Outputs

    %% Styling for clarity
    classDef input fill:#f9f,stroke:#333,stroke-width:2px;
    classDef gate fill:#bbf,stroke:#333,stroke-width:2px;
    classDef operation fill:#bfb,stroke:#333,stroke-width:2px;
    class xt,ht_prev,ct_prev input;
    class z,ft,it,gt,ot,zf,zi,zg,zo gate;
    class ct,tanh_ct,ht operation;
```

$$z = x_t \cdot W + h_{t-1} \cdot U + b$$

$$f_t = \sigma(z_f), z_f = x_t \cdot W_f + h_{t-1} \cdot U_f + b_f$$
$$i_t = \sigma(z_i)$$
$$g_t = \tanh(z_g)$$
$$o_t = \sigma(z_o)$$

$$c_t = f_t \cdot c_{t-1} + i_t \cdot g_t$$
$$h_t = o_t \cdot \tanh(c_t)$$

$$\hat{y}_{pred} = \text{Linear}(h_TW_y + b_y)$$



```mermaid
graph TD
    subgraph Input
        A["x<sub>t</sub>"]
        B["h<sub>t-1</sub>"]
        C["c<sub>t-1</sub>"]
    end
    subgraph LSTM Cell
        D["z = x<sub>t</sub>W + h<sub>t-1</sub>U + b<br>z = [z<sub>f</sub> | z<sub>i</sub> | z<sub>g</sub> | z<sub>o</sub>]"]
        D --> E["z<sub>o</sub>"]
        D --> F["z<sub>f</sub>"]
        D --> G["z<sub>g</sub>"]
        D --> H["z<sub>i</sub>"]
        E --> I["o<sub>t</sub> = σ(z<sub>o</sub>)"]
        F --> J["f<sub>t</sub> = σ(z<sub>f</sub>)"]
        G --> K["g<sub>t</sub> = tanh(z<sub>g</sub>)"]
        H --> L["i<sub>t</sub> = σ(z<sub>i</sub>)"]
        J --> M["c<sub>t</sub> = f<sub>t</sub> · c<sub>t-1</sub> + i<sub>t</sub> · g<sub>t</sub>"]
        K --> M
        L --> M
        M --> N["h<sub>t</sub> = o<sub>t</sub> · tanh(c<sub>t</sub>)"]
        I --> N
    end
    A --> D
    B --> D
    C --> M

```


```mermaid
graph TD
    %% Data Preparation
    subgraph Data_Preparation
        A[Load Dataset] --> B[Preprocess Data]
        B --> C[Create Training & Test Sets]
        C --> D[X_train, Y_train]
        C --> E[X_test, Y_test]
    end

    %% Training Loop
    subgraph Training_Loop["Training Loop (Epochs)"]
        F[Initialize Parameters] --> G[For Each Epoch]
        G --> H[Forward Pass]
        H --> I[Compute Loss]
        I --> J[Backward Pass]
        J --> K[Compute Gradients]
        K --> L[Update Parameters]
        L --> M[Next Epoch]
        M --> G
    end

    %% Components
    F -.-> N[LSTMNetwork Initialization]
    H -.-> O[Input: X_train]
    I -.-> P[Loss Function: MSE]
    J -.-> Q[Backpropagation Through Time]
    L -.-> R[Optimizer: AdamLSTM]
    
    %% After Training
    G --> S[End of Training]
    S --> T[Model Evaluation]
    T --> U[Make Predictions on X_test]
    U --> V[Plot Actual vs Predicted]
    
    %% Styling
    classDef data fill:#f9f,stroke:#333,stroke-width:2px;
    classDef process fill:#bbf,stroke:#333,stroke-width:2px;
    classDef component fill:#bfb,stroke:#333,stroke-width:2px;
    class A,B,C,D,E data;
    class F,G,H,I,J,K,L,M,S,T,U,V process;
    class N,O,P,Q,R component;
```


```mermaid
flowchart TD
    %% Forward Pass
    subgraph Forward Pass
        X0["Input x₀"] -->|Compute| H0["Hidden h₀"]
        H0 -->|Compute| H1["Hidden h₁"]
        H1 -->|Compute| H2["Hidden h₂"]
        H2 -->|...| HT["Hidden h_T"]
        HT -->|Output| Ypred["Y_pred"]
    end

    %% Loss Computation
    Ypred -->|Compute Loss| Loss["Loss L"]

    %% Backward Pass
    subgraph Backward Pass
        Loss -->|dL/dY| dY["dY"]
        dY -->|Gradients| dWy["dW_y"]
        dY -->|Gradients| dby["dB_y"]
        dY -->|Backpropagate| dhT["dh_T"]
        dhT -->|Through Output Gate| doT["do_T"]
        dhT -->|Through Cell State| dcT["dc_T"]
        doT -->|Concatenate| dzT["dz_T"]
        dcT -->|Through Forget Gate| dcTMinus1["dc_{T-1}"]
        dzT -->|Gradients| dW["dW"]
        dzT -->|Gradients| dU["dU"]
        dzT -->|Gradients| db["db"]
        dzT -->|Backpropagate| dhTMinus1["dh_{T-1}"]

        %% Iterative Steps for t = T-1 to 1
        dhTMinus1 -->|Through Output Gate| doTMinus1["do_{T-1}"]
        dhTMinus1 -->|Through Cell State| dcTMinus1Sub["dc_{T-2}"]
        doTMinus1 -->|Concatenate| dzTMinus1["dz_{T-1}"]
        dzTMinus1 -->|Gradients| dW
        dzTMinus1 -->|Gradients| dU
        dzTMinus1 -->|Gradients| db
        dzTMinus1 -->|Backpropagate| dhTMinus2["dh_{T-2}"]

        %% Continue Backpropagation until t=1
        subgraph Time Steps
            direction TB
            dhTMinus2 -->|Through Output Gate| doTMinus2["do_{T-2}"]
            doTMinus2 -->|Concatenate| dzTMinus2["dz_{T-2}"]
            dzTMinus2 -->|Gradients| dW
            dzTMinus2 -->|Gradients| dU
            dzTMinus2 -->|Gradients| db
            dzTMinus2 -->|Backpropagate| dhTMinus3["dh_{T-3}"]
            %% ... continue as needed ...
            dhTMinus3 -->|Through Output Gate| doTMinus3["do_{T-3}"]
            doTMinus3 -->|Concatenate| dzTMinus3["dz_{T-3}"]
            dzTMinus3 -->|Gradients| dW
            dzTMinus3 -->|Gradients| dU
            dzTMinus3 -->|Gradients| db
            dzTMinus3 -->|Backpropagate| dhTMinus4["dh_{T-4}"]
        end
    end

```


```mermaid
graph TD
    %% Start from Loss
    Loss["Loss"] --> dY["dL/dY_pred"]

    %% Output Layer Gradients
    dY --> dWy["dL/dW<sub>y</sub> = h<sub>T</sub><sup>T</sup> ⋅ dY"]
    dY --> dby["dL/db<sub>y</sub> = Σ(dY)"]
    dY --> dhT["dL/dh<sub>T</sub> = dY ⋅ W<sub>y</sub><sup>T</sup>"]

    %% Backpropagation Through Time
    dhT --> BPTT["Backpropagation Through Time (BPTT)"]

    subgraph BPTT
        direction TB
        %% Time step t (from T to 1)
        TimeStep["Time step t"]

        %% Gradients w.r. to gates
        TimeStep --> do_t["do<sub>t</sub> = dh<sub>t</sub> * tanh(c<sub>t</sub>) * o<sub>t</sub> * (1 - o<sub>t</sub>)"]
        TimeStep --> dc_t["dc<sub>t</sub> = dh<sub>t</sub> * o<sub>t</sub> * (1 - tanh^2(c<sub>t</sub>))"]

        dc_t --> df_t["df<sub>t</sub> = dc<sub>t</sub> * c<sub>t-1</sub> * f<sub>t</sub> * (1 - f<sub>t</sub>)"]
        dc_t --> di_t["di<sub>t</sub> = dc<sub>t</sub> * g<sub>t</sub> * i<sub>t</sub> * (1 - i<sub>t</sub>)"]
        dc_t --> dg_t["dg<sub>t</sub> = dc<sub>t</sub> * i<sub>t</sub> * (1 - g<sub>t</sub>^2)"]

        %% Combine gate gradients
        df_t --> dz_t["dz<sub>t</sub> = [df<sub>t</sub>, di<sub>t</sub>, dg<sub>t</sub>, do<sub>t</sub>]"]
        di_t --> dz_t
        dg_t --> dz_t
        do_t --> dz_t

        %% Accumulate parameter gradients
        dz_t --> dW["dW += x<sub>t</sub><sup>T</sup> ⋅ dz<sub>t</sub>"]
        dz_t --> dU["dU += h<sub>t-1</sub><sup>T</sup> ⋅ dz<sub>t</sub>"]
        dz_t --> db["db += Σ(dz<sub>t</sub>)"]

        %% Gradients w.r. to previous hidden and cell states
        dz_t --> dh_prev["dh<sub>t-1</sub> = dz<sub>t</sub> ⋅ U<sup>T</sup>"]
        dc_t --> dc_prev["dc<sub>t-1</sub> = dc<sub>t</sub> ⋅ f<sub>t</sub>"]

        %% Loop back for the next (previous) time step
        dh_prev --> TimeStep
        dc_prev --> TimeStep

        %% Indicate repetition
        TimeStep -->|Repeat for t = T-1, T-2, ..., 1| TimeStep
    end

    %% After BPTT, update parameters
    BPTT --> Update["Update Parameters (W, U, b, W<sub>y</sub>, b<sub>y</sub>)"]



    class Loss,dY,dWy,dby,dhT gradient;
    class BPTT,TimeStep,do_t,dc_t,df_t,di_t,dg_t,dz_t,dW,dU,db,dh_prev,dc_prev operation;
    class Update gradient;
```
