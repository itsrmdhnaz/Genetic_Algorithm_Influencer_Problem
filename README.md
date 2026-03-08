# Genetic Algorithm - Optimasi Pemilihan Influencer

Program ini mengimplementasikan Genetic Algorithm (GA) untuk menyelesaikan masalah optimasi pemilihan influencer dengan budget constraint. Tujuannya adalah memaksimalkan total jangkauan followers dengan anggaran maksimal Rp 50 Juta.

## 📋 Deskripsi Masalah

Sebuah perusahaan memiliki database 20 calon influencer untuk mempromosikan produk baru. Setiap influencer memiliki:
- **Tarif** (biaya untuk menyewa influencer)
- **Jangkauan Followers** (jumlah followers yang dapat dijangkau)

**Constraint:**
- Anggaran Maksimal: Rp 50 Juta
- Tujuan: Maksimalkan total followers tanpa melebihi budget

## 🧬 Model Genetic Algorithm

### Representasi Kromosom
- **Tipe Encoding:** Binary Encoding
- **Gen:** 1 = influencer dipilih, 0 = tidak dipilih
- **Panjang Kromosom:** Sesuai jumlah influencer (default: 20)

### Fungsi Fitness
```
Fitness = Total Followers - Penalty
```
- **Penalty:** Soft constraint untuk budget violation
- **Penalty = 1000 × (Total Cost - Max Budget)** jika melebihi budget

### Operator Genetik

1. **Selection:** Tournament Selection (size=3)
2. **Crossover:** 
   - Single-point crossover
   - Multi-point (2-point) crossover
   - Probability-based (50% single, 50% multi)
3. **Mutation:** Bit-Flip Mutation
4. **Survivor Selection:** Elitism (mempertahankan individu terbaik)

## 🚀 Cara Instalasi

### Prerequisites
- Python 3.7 atau lebih tinggi
- pip (Python package manager)

### Langkah Instalasi

1. **Clone atau download repository ini**

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

atau install manual:
```bash
pip install numpy matplotlib
```

## 💻 Cara Menjalankan Program

### Menjalankan GUI Application

```bash
python gui_app.py
```

### Menjalankan Test CLI (tanpa GUI)

```bash
python influencer_ga.py
```

## 🎮 Panduan Penggunaan GUI

### 1. Panel Kiri - Controls

#### Data Generation Section:
1. **Total Influencers (1-100):** Tentukan jumlah influencer yang akan di-generate
   - Default: 20
2. **Seed:** Seed untuk random number generator (opsional)
   - Untuk reproducibility hasil
   - Default: 42
3. **Generate Data:** Klik untuk membuat data influencer

#### GA Parameters:
1. **Population Size (10-500):** Ukuran populasi
   - Default: 50
   - Semakin besar = eksplorasi lebih luas, tapi lebih lambat
2. **Mutation Rate (0.001-1.0):** Probabilitas mutasi per gen
   - Default: 0.01 (1%)
   - Nilai kecil = eksploitasi, nilai besar = eksplorasi
3. **Elitism Count (0-50):** Jumlah individu terbaik yang dipertahankan
   - Default: 2
   - Memastikan solusi terbaik tidak hilang
4. **Crossover Type:** Jenis crossover
   - **single:** Single-point crossover
   - **multi:** Multi-point (2-point) crossover
   - **probability:** Campuran (50% single, 50% multi)
5. **Max Generations (1-1000):** Maksimal generasi yang akan dijalankan
   - Default: 100

#### Control Buttons:
- **Run GA:** Mulai menjalankan algoritma genetika
- **Pause:** Jeda eksekusi (dapat dilanjutkan dengan Resume)
- **Stop:** Hentikan eksekusi sepenuhnya

### 2. Panel Kanan Atas

#### Performance Visualization:
- **Graph 1 (Atas):** Fitness Evolution
  - Garis biru: Best Fitness per generasi
  - Garis merah putus-putus: Average Fitness per generasi
- **Graph 2 (Bawah):** Best Solution Metrics
  - Garis hijau: Total Followers dari solusi terbaik
  - Garis orange: Total Cost dari solusi terbaik

#### Influencer Data Table:
- Menampilkan semua influencer yang di-generate
- Kolom: ID, Name, Tarif (Juta), Followers

### 3. Panel Kanan Bawah

#### Solution Details:
Menampilkan detail solusi terbaik saat ini:
- Generation number
- Seed yang digunakan
- Parameter GA
- Metrics:
  - Jumlah influencer terpilih
  - Total followers
  - Total cost
  - Budget usage percentage
  - Fitness score
  - Penalty
- List influencer yang terpilih
- Representasi kromosom (binary string)

#### Execution Log:
- Menampilkan log eksekusi program
- Update setiap 10 generasi
- Informasi error dan warning

## 📊 Contoh Workflow

1. **Set Total Influencers = 20**
2. **Set Seed = 42** (untuk hasil konsisten)
3. **Klik "Generate Data"**
   - Data influencer akan muncul di tabel
4. **Set GA Parameters:**
   - Population Size = 50
   - Mutation Rate = 0.01
   - Elitism Count = 2
   - Crossover Type = probability
   - Max Generations = 100
5. **Klik "Run GA"**
   - GA akan berjalan otomatis
   - Graph akan update real-time
   - Log akan menampilkan progress
6. **Monitor hasil:**
   - Perhatikan fitness evolution di graph
   - Lihat solution details untuk solusi terbaik
7. **Pause/Resume jika diperlukan**
8. **Tunggu hingga selesai atau Stop manual**

## 🏗️ Struktur Kode (OOP Design)

### File: `influencer_ga.py`

#### Class: `Influencer`
- Data class untuk representasi influencer
- Attributes: id, name, tarif, followers

#### Class: `Individual`
- Representasi satu solusi (kromosom)
- Methods:
  - `calculate_fitness()`: Hitung fitness dengan penalty
  - `get_selected_influencers()`: Dapatkan influencer terpilih

#### Class: `GeneticAlgorithm`
- Implementasi algoritma genetika
- Methods:
  - `initialize_population()`: Inisialisasi populasi
  - `tournament_selection()`: Seleksi parent
  - `crossover()`: Operasi crossover
  - `mutate()`: Operasi mutasi
  - `evolve()`: Evolusi satu generasi
  - `get_best_solution()`: Dapatkan solusi terbaik

#### Function: `generate_influencer_data()`
- Generate data dummy influencer
- Support custom seed untuk reproducibility

### File: `gui_app.py`

#### Class: `InfluencerGAApp`
- Main GUI application menggunakan tkinter
- Components:
  - Input controls (left panel)
  - Visualization dengan matplotlib
  - Data table dengan ttk.Treeview
  - Solution details
  - Execution log
- Multi-threading untuk non-blocking UI

## 🔧 Best Practices yang Diimplementasikan

### 1. Object-Oriented Programming (OOP)
- Encapsulation: Data dan method dalam class
- Separation of Concerns: GA logic terpisah dari UI
- Single Responsibility: Setiap class punya tanggung jawab jelas

### 2. Error Handling
- Try-except blocks di semua critical operations
- Graceful error messages untuk user
- Validation input parameters

### 3. Code Quality
- Type hints untuk parameter dan return values
- Comprehensive docstrings
- Descriptive variable names
- Modular function design

### 4. User Experience
- Real-time visualization updates
- Responsive UI dengan threading
- Clear progress logging
- Pause/Resume functionality

## 📈 Parameter Tuning Tips

### Population Size
- **Kecil (10-30):** Cepat, tapi mungkin terjebak di local optimum
- **Sedang (50-100):** Balance antara kecepatan dan kualitas
- **Besar (200+):** Eksplorasi lebih baik, tapi lambat

### Mutation Rate
- **Rendah (0.001-0.01):** Untuk fine-tuning solusi
- **Sedang (0.01-0.05):** Balance eksplorasi-eksploitasi
- **Tinggi (0.1+):** Untuk problem yang sulit atau escape local optimum

### Elitism Count
- **Rendah (1-2):** Cukup untuk menjaga solusi terbaik
- **Sedang (5-10):** Lebih stabil tapi kurang diverse
- **Tinggi:** Tidak direkomendasikan (membatasi eksplorasi)

### Crossover Type
- **Single:** Simple, cepat, good for most cases
- **Multi:** Lebih eksplorasi, bagus untuk problem kompleks
- **Probability:** Balance, recommended untuk general use

## 🐛 Troubleshooting

### Program tidak jalan
```bash
# Check Python version
python --version

# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

### GUI tidak muncul atau error
```bash
# Pada macOS, mungkin perlu install tkinter
brew install python-tk

# Pada Linux
sudo apt-get install python3-tk
```

### Matplotlib error
```bash
# Install ulang matplotlib
pip uninstall matplotlib
pip install matplotlib
```

## 📝 Catatan Penting

1. **Random Seed:** Gunakan seed yang sama untuk hasil yang reproducible
2. **Budget Violation:** Jika best solution masih violate budget, coba:
   - Increase population size
   - Increase max generations
   - Adjust penalty coefficient (di code)
3. **Performance:** Untuk dataset besar (>50 influencer), pertimbangkan:
   - Reduce max generations
   - Reduce population size sedikit
   - Use single-point crossover

## 📜 License

Program ini dibuat untuk keperluan educational/assignment.

## 👨‍💻 Author

Dibuat dengan ❤️ menggunakan Python, tkinter, dan matplotlib.

---

**Selamat menggunakan! Jika ada pertanyaan atau issue, silakan hubungi developer.**
