import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

def generate_overfitting_loss(epochs=50):
    x = np.arange(1, epochs + 1)
    
    # Eğitim kaybı (Train Loss) sürekli azalır
    train_loss = 0.05 * np.exp(-x / 8.0) + 0.002 + np.random.normal(0, 0.0003, size=epochs)
    
    # Doğrulama kaybı (Validation Loss) başta azalır, sonra artmaya başlar (U şeklinde)
    # Optimum nokta yaklaşık 15-20. epochlar arası
    val_loss = 0.048 * np.exp(-x / 6.0) + 0.004 + 0.000015 * (x - 18)**2 + np.random.normal(0, 0.0004, size=epochs)
    
    train_loss = np.maximum(train_loss, 0.001)
    val_loss = np.maximum(val_loss, 0.001)
    
    return train_loss, val_loss

def main():
    epochs = 50
    train_losses, val_losses = generate_overfitting_loss(epochs)
    
    plt.figure(figsize=(10, 6))
    
    # Plotting
    plt.plot(range(1, epochs + 1), train_losses, label='Eğitim Kaybı (Train Loss)', color='#2E86C1', linewidth=2.5)
    plt.plot(range(1, epochs + 1), val_losses, label='Doğrulama Kaybı (Validation Loss)', color='#E74C3C', linewidth=2.5, linestyle='--')
    
    # Erken Durdurma (Early Stopping) noktasını işaretleme
    optimal_epoch = np.argmin(val_losses) + 1
    optimal_loss = val_losses[optimal_epoch - 1]
    
    plt.axvline(x=optimal_epoch, color='#27AE60', linestyle=':', linewidth=2, label='Erken Durdurma (Early Stopping) Noktası')
    plt.scatter(optimal_epoch, optimal_loss, color='#27AE60', s=100, zorder=5)
    
    # Aşırı öğrenme (Overfitting) bölgesini vurgulama
    plt.axvspan(optimal_epoch, epochs, color='#FADBD8', alpha=0.3, label='Aşırı Öğrenme (Overfitting) Bölgesi')
    
    # Kural: Grafiğin içine başlık yazmıyoruz. 
    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    plt.ylabel('Ortalama Kare Hatası (MSE)', fontsize=12, fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.7)
    
    # Lejant
    plt.legend(fontsize=11, loc='upper center', frameon=True, shadow=True)
    
    # Eksenleri biraz daha şık yapalım
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    output_dir = 'dosyalar'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'Sekil_5_20_Overfitting.png')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"Grafik basariyla kaydedildi: {output_path}")

if __name__ == '__main__':
    main()
