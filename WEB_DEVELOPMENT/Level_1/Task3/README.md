# Task 3 — Interactive Calculator Engine 🧮

> **CodSoft Web Development Internship | May Batch C1 ID:BY25RY291730**  
> **Author : Yashwanth G S**  
> **GitHub : https://github.com/Yashwanth181024**  

---

## 📌 Objective

Build a modern, interactive, and fully functional calculator interface that handles arithmetic operations seamlessly using clean, plagiarism-safe JavaScript implementation.

---

## 📂 Project Structure

| File / Folder | Description |
|---|---|
| `index.html` | Structural foundation defining the display monitor screen and responsive button grid |
| `style.css` | Sleek dark UI styling sheet featuring hover micro-animations and custom grid cell layouts |
| `script.js` | Core interactive logic file processing input sequences, operator states, and calculations |

---

## ⚙️ Layout Architecture

| Design Component | Description |
|---|---|
| Button Grid System | Structured using explicit multi-row and multi-column grid layouts to align numerical and operational keys |
| Screen Viewport | A top-aligned, read-only calculation display bar handling real-time input monitoring |
| Alignment Safety | Styled with container tracking properties to ensure no key clusters overlap or break layout lines on smaller viewports |
| Dark UI Palette | Uses high-legibility contrasting elements to isolate operational triggers from numerical entries |

---

## 🔧 Logic Processing & Edge Cases

| Action Type | Handling Routine |
|---|---|
| Input Accumulation | Appends numerical keys sequentially while ensuring multi-decimal inputs are blocked |
| Operation Routing | Processes standard operator inputs (`+`, `-`, `*`, `/`) dynamically inside conditional branches |
| Clear Command (`C`) | Flushes active tracking registers and resets the live UI monitor display value to zero |
| Calculation Execution | Evaluates the string expression through safe, custom conditional state steps rather than shorthand flags |
| Error Safeguards | Features runtime error handling exceptions to cleanly intercept invalid operations like division-by-zero |

---

## 🚀 How to Run

```bash
# Navigate directly to your interactive calculator directory
cd CODSOFT/WEB_DEVELOPMENT/Level_1/Task3_Calculator

# Open the main execution layout document in your web browser
start index.html

```

---

## 📈 Key Layout Features

* **No Evaluation Shortcuts:** Built safely by avoiding the heavily flagged `eval()` pattern entirely. This script tracks operations natively through distinct mathematical logic routines.
* **Dynamic Feedback:** Enhanced with intuitive, immediate background transitions when a user clicks on numerical or operator keys.
* **Fluid Layout Framework:** Employs precise alignment padding properties to guarantee standard presentation looks clean across varying web dimensions.

---

## 🛠️ Stack Elements

```
HTML5 | CSS3 | Vanilla JavaScript | Grid Mapping Layouts

```

---

## 👤 Author

**Yashwanth G S**

GitHub : https://github.com/Yashwanth181024

Internship : CodSoft Web Development Internship | May Batch C1 ID:BY25RY291730

```

```
