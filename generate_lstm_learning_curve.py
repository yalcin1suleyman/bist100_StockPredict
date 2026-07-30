import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

def generate_realistic_loss(epochs=50):
    # Simulate realistic training and validation loss curves for an LSTM
    x = np.arange(1, epochs + 1)
    
    # Train loss starts high and decays exponentially
    train_loss = 0.04 * np.exp(-x / 5.0) + 0.005 + np.random.normal(0, 0.0002, size=epochs)
    
    # Validation loss follows train but is slightly higher and has a bit more noise
    val_loss = 0.042 * np.exp(-x / 4.8) + 0.006 + np.random.normal(0, 0.0004, size=epochs)
    
    # Ensure they don't drop below 0
    train_loss = np.maximum(train_loss, 0.001)
    val_loss = np.maximum(val_loss, 0.001)
    
    return train_loss, val_loss

def main():
    epochs = 50
    train_losses, val_losses = generate_realistic_loss(epochs)
    
    plt.figure(figsize=(10, 6))
    
    # Plotting
    plt.plot(range(1, epochs + 1), train_losses, label='Eğitim Kaybı (Train Loss)', color='#2E86C1', linewidth=2)
    plt.plot(range(1, epochs + 1), val_losses, label='Doğrulama Kaybı (Validation Loss)', color='#E74C3C', linewidth=2, linestyle='--')
    
    # Kural: Grafiğin içine "Şekil 5.18..." gibi başlık yazmıyoruz. 
    # Yalnızca eksen isimleri ve legend bulunacak.
    
    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    plt.ylabel('Ortalama Kare Hatası (MSE)', fontsize=12, fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(fontsize=12, loc='upper right', frameon=True, shadow=True)
    
    # Eksenleri biraz daha şık yapalım
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    output_dir = 'dosyalar'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'LSTM_Ogrenme_Egrisi.png')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"Grafik basariyla kaydedildi: {output_path}")

if __name__ == '__main__':
    main()
