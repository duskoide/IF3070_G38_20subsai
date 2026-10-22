Tugas Besar 1 IF3070 Dasar Inteligensi Artifisial 2026/2027

Program Python untuk menentukan kendaraan yang diangkut beserta posisi dan orientasinya di geladak kapal. Objectivenya adalah untuk memaksimalkan total tarif pengiriman (*ShippingFee*) tanpa kendaraan overlapping, overlapping, atau overweight kapasitas kapal.

Algoritma yang diimplementasikan:
- Hill-climbing: Steepest Ascent, Sideways Move, Random Restart, dan Stochastic.
- Simulated Annealing.
- Genetic Algorithm.

## Setup

Siapkan Python 3.11+ dan uv (atau pip), lalu jalankan dari root directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Menjalankan Program

```bash
python -m src.main
```

atau dengan uv:

```
uv run -m src.main
```

1. Pilih algoritma, atau pilih `1` untuk menjalankan semua algoritma.
2. Masukkan seed dan parameter yang diminta. Tekan Enter untuk memakai nilai default.
3. Masukkan `y` pada pertanyaan `Simpan visualisasi?` untuk menyimpan gambar state awal, state akhir, dan grafik pencarian ke `output/`.
4. Hasil pencarian ditampilkan di terminal, termasuk nilai objective, kelayakan solusi, jumlah iterasi, dan durasi. Pilih `0` untuk keluar.

Data input dibaca dari `src/data/sample.json`, yang berisi dimensi dan kapasitas kapal serta daftar kendaraan.

## Pembagian Tugas

Lengkapi tabel berikut sesuai identitas dan kontribusi sebenarnya.

| Nama | NIM | Tugas |
|---|---|---|
| Yumna Fathonah Kautsar | 18224063 | Mengerjakan Steepest Ascent Hill-Climbing Algorithm, mengerjakan Random Restart Hill-Climbing Algorithm, mengerjakan Stochastic Hill-Climbing Algorithm, mengerjakan Sideways Move Hill-Climbing Algorithm |
| Rafi Putra Nugraha | 18224079 | Mengerjakan Genetic Algorithm, membuat model.py, loader.py, visualicz dan objective.py |
| Illona Nasywa Hannum | 18224079 | Mengerjakan Simulated Annealing Algorithm dan neighbors.py |
