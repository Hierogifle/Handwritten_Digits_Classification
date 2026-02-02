# Architecture de l'application

````mermaid
graph LR
    subgraph NET["🌐 Accès Réseau"]
        A["💻 Local<br/>10.10.98.111:5000"]
        B["📱 Réseau Distant<br/>HTTPS ngrok"]
    end
    
    subgraph UI["🎨 Interface Web"]
        C["🏠 Page Principale<br/>index.html"]
    end
    
    subgraph MODES["📋 Modes de Saisie"]
        D["📤 Upload<br/>Fichier Image"]
        E["✏️ Canvas<br/>Dessin Manuel"]
        F["📸 Caméra<br/>Capture Photo"]
    end
    
    subgraph PROC["⚙️ Pipeline de Traitement"]
        G["🔍 Validation<br/>Format & Taille"]
        I["💾 Stockage<br/>Temporaire"]
        H["🖼️ Preprocessing<br/>28x28 Grayscale"]
    end
    
    subgraph AI["🤖 Modèles PyTorch"]
        J["⚡ Device<br/>GPU/CPU"]
        K["🧠 MLP<br/>160-112"]
        L["🔬 CNN<br/>Conv 64-64"]
    end
    
    subgraph RES["📊 Résultat"]
        M["✅ Réponse JSON<br/>2 Prédictions"]
    end
    
    A --> C
    B --> C
    C --> D
    C --> E
    C --> F
    
    D --> G
    E --> G
    F --> G
    
    G --> I
    I --> H
    H --> J
    J --> K
    J --> L
    K --> M
    L --> M
    M --> C
    
    I -.->|🗑️ Auto-suppression| N["Cleanup"]
    
    style C fill:#4CAF50,stroke:#2E7D32,stroke-width:2px,color:#fff
    style K fill:#2196F3,stroke:#1565C0,stroke-width:2px,color:#fff
    style L fill:#2196F3,stroke:#1565C0,stroke-width:2px,color:#fff
    style M fill:#00BCD4,stroke:#006064,stroke-width:2px,color:#fff
    style I fill:#FF9800,stroke:#E65100,stroke-width:2px,color:#fff
    style B fill:#9C27B0,stroke:#4A148C,stroke-width:2px,color:#fff
    style G fill:#F44336,stroke:#C62828,stroke-width:2px,color:#fff
````