# 🌎 Genetic TSP Optimizer

This project solves the **Traveling Salesman Problem (TSP)** using a **Genetic Algorithm (GA)** enhanced with:

* **Nearest Neighbor Heuristic (NNH)** seeding
* **Adaptive mutation** for diversity
* **Elitism** for preserving top-performing solutions

The algorithm dynamically adjusts configurations based on input size and supports `.txt` formatted data, making it flexible and efficient for various TSP scales.

---

## 🚀 Features

* ✅ Nearest Neighbor Heuristic (NNH)-seeded initial population
* 🔁 Two-point crossover with duplicate prevention
* 🧬 Adaptive mutation rate scaled over generations
* 🏆 Elitism: best individuals retained across generations
* 🎯 Roulette Wheel Selection using normalized fitness
* 📏 Euclidean distance-based scoring
* ⚙️ Parameter tuning based on input size
* 📄 Readable input/output using text files

---

## 🧰 How It Works (Step-by-Step)

### 1. 📥 Input File

Reads `input.txt`, which contains the number of cities and their `(x, y, id)` coordinates.

### 2. 📐 Distance Matrix

Builds a full Euclidean distance matrix between all cities.

### 3. 🌱 Population Initialization

Mixes:

* 60% from Nearest Neighbor tours
* 40% random permutations (diversity boost)

### 4. 📊 Fitness Evaluation

Each route is scored based on total distance (lower = better). Fitness scores are normalized.

### 5. 🎰 Selection

Roulette Wheel Selection probabilistically picks parents using normalized fitness scores.

### 6. 🔀 Crossover

Two-point crossover builds children from parents while maintaining valid city sequences.

### 7. 🧪 Mutation

Adaptive mutation reverses random route segments with a decreasing mutation rate.

### 8. 🥇 Elitism

Top-performing individuals are preserved into the next generation.

### 9. 🛑 Early Stopping

If the best route doesn't improve for `n` generations, evolution halts early.

### 10. 📤 Output

Saves the best route and its total distance to `output.txt`.

---

## 📁 Sample Input (`input.txt`)

```txt
5
0 0 0
1 2 1
3 1 2
6 0 3
5 5 4
```

* Line 1: Number of cities
* Line 2+: Coordinates and index of each city

---

## 📄 Sample Output (`output.txt`)

```txt
12.345
0 0 0
1 2 1
...
```

* Line 1: Total distance of best route
* Line 2+: Cities in the best visiting sequence

---

## 🧪 How to Run

1. Ensure `input.txt` is in the project folder.
2. Execute the script:

```bash
python tsp_genetic_optimizer.py
```

3. View results in:

```bash
output.txt
```

---

## 🗂️ Suggested Repo Name

> **`genetic-tsp-optimizer`**

---

## 🛠 Technologies Used

* 🐍 Python 3.9+
* 🧮 NumPy (distance matrix, mutation, selection)
* 🎲 Built-in `random`, file I/O

---

## 👤 Author

**Rohit Ramesh Shetty**
MS CS @ USC | Optimization & AI Enthusiast
[🔗 GitHub](https://github.com/shettyrohit0810) ・ [🔗 LinkedIn](https://linkedin.com/in/shettyrohit0810)

---

> ⭐ *Star this repo if it helped you! Contributions and feedback are welcome.*
