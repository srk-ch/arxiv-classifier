# 🎓 arXiv Paper Classifier

An AI-powered web application that automatically classifies research papers into academic categories using machine learning. Built with Flask and CatBoost, achieving 87.56% classification accuracy.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)
![CatBoost](https://img.shields.io/badge/CatBoost-ML-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## ✨ Features

- **AI-Powered Classification**: Uses CatBoost machine learning model with 87.56% accuracy
- **5 Academic Categories**: AI/ML, Physics, Mathematics, Biology & Health, Chemistry & Materials
- **User Authentication**: Secure signup/login with email verification
- **Classification History**: Track all your paper classifications
- **Admin Dashboard**: Comprehensive user management and analytics
- **Confidence Scoring**: View top 5 predictions with confidence percentages
- **Beautiful UI**: Modern, responsive design with gradient effects and animations
- **Email Notifications**: Automated verification emails via Flask-Mail

## 🚀 Demo

### Main Classifier Interface
- Input paper title and abstract
- Get instant classification results
- View confidence scores and top predictions
- Access classification history

### Admin Panel
- View all users and their statistics
- Monitor total classifications
- View user-specific classification history
- Delete users (non-admin only)

## 🛠️ Tech Stack

### Backend
- **Python 3.8+**
- **Flask** - Web framework
- **SQLAlchemy** - ORM for database management
- **Flask-Mail** - Email verification system
- **CatBoost** - Machine learning model
- **Scikit-learn** - Feature extraction (TF-IDF)
- **Joblib** - Model serialization

### Frontend
- **HTML5/CSS3** - Modern, responsive design
- **JavaScript (Vanilla)** - Dynamic interactions
- **Google Fonts** - Typography (Space Grotesk, Inter)

### Database
- **SQLite** - Lightweight database for development

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Gmail account (for email verification)

## 🔧 Installation

### Option 1: Use Pre-trained Model (Quick Start)

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/arxiv-classifier.git
cd arxiv-classifier
```

2. **Create virtual environment**
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install flask flask-sqlalchemy flask-mail catboost scikit-learn joblib numpy scipy werkzeug
```

4. **Download pre-trained model files**

The following files should be in the root directory:
- `catboost_hybrid.cbm` - Trained CatBoost model
- `vectorizer.pkl` - TF-IDF vectorizer
- `keywords.pkl` - Domain-specific keywords

5. **Configure email settings**

Edit `app.py` and update email configuration:
```python
app.config['MAIL_USERNAME'] = 'your-email@gmail.com'
app.config['MAIL_PASSWORD'] = 'your-app-password'
app.config['MAIL_DEFAULT_SENDER'] = 'your-email@gmail.com'
```

**Note**: For Gmail, you need to generate an [App Password](https://support.google.com/accounts/answer/185833).

6. **Initialize database**
```bash
python app.py
```

This will create the SQLite database and default admin account.

### Option 2: Train Your Own Model (Advanced)

Want to improve accuracy or train on your own dataset? Follow these steps:

1. **Open the training notebook**

Access the Google Colab notebook for model training:
```
📓 [Model Training Notebook](https://colab.research.google.com/drive/1EKN2jFlSww5aK7DFrfGf2l_8A3jGyB0l)
```

2. **Prepare your data**

The notebook uses the arXiv metadata snapshot. You can:
- Use the default arXiv dataset (15,000 papers, 3,000 per category)
- Increase `target_per_class` in the notebook to train on more samples
- Or provide your own dataset in the required format

**Dataset format** (if using custom data):
```python
# JSON format (one paper per line)
{
  "title": "Paper title here",
  "abstract": "Abstract text here",
  "categories": "cs.LG stat.ML"
}
```

3. **Run the training notebook**

The notebook consists of 3 main blocks:

**Block 1 - Setup** (30 seconds)
- Mount Google Drive
- Install dependencies (CatBoost, scikit-learn, etc.)

**Block 2 - Dataset Creation** (2-3 minutes)
- Loads arXiv metadata
- Creates balanced dataset (3,000 papers per category)
- Cleans text (removes LaTeX, mathematical notation)
- Splits into train/validation/test sets
- Current categories:
  - AI & ML: cs.LG, cs.AI, cs.CL, cs.CV, cs.NE, cs.RO, stat.ML
  - Physics: hep-ph, hep-th, astro-ph, gr-qc, quant-ph, nucl-th, nucl-ex
  - Mathematics: math.AG, math.AT, math.CO, math.DG, math.NT, math.PR, math.ST
  - Biology & Health: q-bio.BM, q-bio.GN, q-bio.NC, q-bio.QM, q-bio.SC
  - Chemistry & Materials: cond-mat.mtrl-sci, cond-mat.str-el, etc.

**Block 3 - Model Training** (3-4 minutes)
- Hybrid approach: TF-IDF (15,000 features) + domain-specific keywords
- CatBoost classifier with optimized hyperparameters
- Early stopping to prevent overfitting
- Automatically saves 3 files:
  - `catboost_hybrid.cbm`
  - `vectorizer.pkl`
  - `keywords.pkl`

**Final Cell - Download Files**
- Downloads all 3 trained files to your computer

4. **Customize training (optional)**

**Increase training samples:**
```python
# In Block 2, change this line:
target_per_class = 3000  # Change to 5000, 10000, etc.
```

**Adjust model parameters:**
```python
# In Block 3, modify CatBoost parameters:
cat = CatBoostClassifier(
    iterations=400,           # More iterations = better accuracy (slower)
    learning_rate=0.15,       # Lower = more careful learning
    depth=6,                  # Deeper trees = more complex patterns
    early_stopping_rounds=50  # Patience before stopping
)
```

**Add custom keywords:**
```python
# In Block 2, extend DOMAIN_KEYWORDS dictionary:
DOMAIN_KEYWORDS = {
    'AI_ML': ['neural', 'network', 'deep', 'learning', 'transformer', ...],
    'Physics': ['quantum', 'particle', 'photon', ...],
    # Add more keywords to improve classification
}
```

5. **Download trained files**

After training completes, the last cell automatically downloads:
- `catboost_hybrid.cbm` (trained model)
- `vectorizer.pkl` (TF-IDF vectorizer)
- `keywords.pkl` (domain keywords)

6. **Replace model files**

Copy the downloaded files to your project root directory:
```bash
# Navigate to your project directory
cd arxiv-classifier

# Backup old models (optional but recommended)
mkdir models_backup
cp catboost_hybrid.cbm vectorizer.pkl keywords.pkl models_backup/

# Copy new trained models from Downloads
cp ~/Downloads/catboost_hybrid.cbm .
cp ~/Downloads/vectorizer.pkl .
cp ~/Downloads/keywords.pkl .
```

7. **Test your new model**
```bash
python app.py
```

Your app now uses your custom-trained model! 🎉

**Expected Training Time:**
- Setup: ~30 seconds
- Dataset creation: 2-3 minutes
- Model training: 3-4 minutes
- **Total: ~6-8 minutes** ⚡

## 🎮 Usage

### Starting the Application

```bash
python app.py
```

The application will be available at `http://127.0.0.1:5000`

### Default Admin Account

- **Username**: `admin`
- **Password**: `admin123`
- **Email**: `admin@example.com`

⚠️ **Important**: Change the admin password after first login!

### User Workflow

1. **Sign Up**: Create an account with username, email, and password
2. **Verify Email**: Check your email for verification link
3. **Login**: Access the classifier interface
4. **Classify Papers**: Enter title and abstract for classification
5. **View Results**: See predictions, confidence scores, and history
6. **Track History**: Access all previous classifications via sidebar

### Admin Workflow

1. **Login** as admin
2. **View Dashboard**: See user statistics and system metrics
3. **Manage Users**: View user details and classification history
4. **Delete Users**: Remove non-admin users if needed

## 📊 Supported Categories

| Category | Icon | Description |
|----------|------|-------------|
| AI & Machine Learning | 🤖 | Artificial intelligence, neural networks, deep learning |
| Physics | ⚛️ | Theoretical, experimental, and applied physics |
| Mathematics | 📐 | Pure and applied mathematics, algebra, geometry |
| Biology & Health | 🧬 | Biological sciences, medical research, health sciences |
| Chemistry & Materials | 🧪 | Chemistry, materials science, related fields |

## 🏗️ Project Structure

```
arxiv-classifier/
│
├── app.py                      # Main Flask application
├── catboost_hybrid.cbm         # Trained ML model
├── vectorizer.pkl              # TF-IDF vectorizer
├── keywords.pkl                # Domain keywords
├── arxiv_classifier.db         # SQLite database
│
├── templates/
│   ├── home.html              # Landing page
│   ├── about.html             # About project page
│   ├── aboutus.html           # Team information
│   ├── signup.html            # User registration
│   ├── login.html             # User login
│   ├── classifier.html        # Main classifier interface
│   └── admin.html             # Admin dashboard
│
├── notebooks/
│   └── model_training.ipynb   # Google Colab notebook for training
│
└── README.md                  # This file
```

## 🧠 Model Training

### Training Your Own Model

We provide a **Google Colab notebook** that trains the model in just **6-8 minutes**! 🚀

**Access the Training Notebook**: [📓 Google Colab Training Notebook](https://colab.research.google.com/drive/1EKN2jFlSww5aK7DFrfGf2l_8A3jGyB0l)

### What the Notebook Does

The notebook automatically:

1. **Block 1 - Setup** (~30 seconds)
   - Mounts Google Drive
   - Installs required packages (CatBoost, scikit-learn, tqdm)
   - Imports necessary libraries

2. **Block 2 - Dataset Creation** (2-3 minutes)
   - Loads arXiv metadata snapshot from Google Drive
   - Builds balanced dataset: **15,000 papers** (3,000 per category)
   - Text preprocessing:
     - Removes LaTeX notation and mathematical symbols
     - Cleans and normalizes text
     - Filters papers with <30 words
   - Splits data: 70% train, 15% validation, 15% test

3. **Block 3 - Hybrid Model Training** (3-4 minutes)
   - **TF-IDF Features**: 15,000 features with bigrams
   - **Keyword Features**: Domain-specific keyword matching
   - **CatBoost Classifier**: Fast gradient boosting
   - **Early Stopping**: Prevents overfitting
   - Automatically saves all 3 model files

4. **Final Cell - Download**
   - One-click download of all trained files

### Training Architecture

**Hybrid Feature Engineering:**
```python
TF-IDF Vectorization (15,000 features)
    +
Domain-Specific Keywords (50+ per category)
    =
Enhanced Classification Accuracy
```

**Categories & Keywords:**

| Category | arXiv Categories | Keywords (examples) |
|----------|------------------|---------------------|
| AI & ML | cs.LG, cs.AI, cs.CV, cs.CL, cs.NE, stat.ML | neural, transformer, attention, lstm, cnn |
| Physics | hep-ph, hep-th, astro-ph, gr-qc, quant-ph | quantum, particle, photon, entanglement |
| Mathematics | math.AG, math.AT, math.CO, math.DG, math.NT | theorem, proof, manifold, topology |
| Biology & Health | q-bio.BM, q-bio.GN, q-bio.NC, q-bio.QM | gene, protein, dna, crispr, genome |
| Chemistry & Materials | cond-mat.mtrl-sci, cond-mat.str-el, cond-mat.soft | crystal, lattice, superconductor, synthesis |

### Customization Options

**1. Increase Training Samples**
```python
# Block 2: Change target_per_class
target_per_class = 3000  # Default
# Change to: 5000, 10000, or more for better accuracy
```

**2. Adjust Model Hyperparameters**
```python
# Block 3: Modify CatBoost settings
cat = CatBoostClassifier(
    iterations=400,           # More = better (but slower)
    learning_rate=0.15,       # Lower = more careful learning
    depth=6,                  # Deeper trees = more complexity
    eval_metric='Accuracy',
    early_stopping_rounds=50
)
```

**3. Tune TF-IDF Features**
```python
# Block 3: Adjust vectorizer
vec = TfidfVectorizer(
    ngram_range=(1,2),        # (1,3) for trigrams
    max_features=15000,       # Increase to 20000, 25000
    min_df=2,                 # Minimum document frequency
    max_df=0.95              # Maximum document frequency
)
```

**4. Add Custom Keywords**
```python
# Block 2: Extend DOMAIN_KEYWORDS
DOMAIN_KEYWORDS = {
    'AI_ML': ['neural', 'network', 'deep', 'learning', 'transformer', 
              'attention', 'lstm', 'cnn', 'rnn', 'gradient'],
    # Add more keywords to improve accuracy
}
```

### Training Performance

**Expected Results:**
- Training Time: 6-8 minutes total
- Final Accuracy: ~87-90% on test set
- Model Size: ~50-100 MB (all 3 files combined)

**Output Files:**
1. `catboost_hybrid.cbm` - Trained CatBoost model
2. `vectorizer.pkl` - TF-IDF vectorizer with vocabulary
3. `keywords.pkl` - Domain-specific keyword dictionary

### Requirements for Training

- Google account (for Colab)
- arXiv metadata snapshot in Google Drive (or download during training)
- ~2 GB RAM (provided free by Colab)
- Internet connection

### Data Source

The notebook uses the official [arXiv metadata snapshot](https://www.kaggle.com/datasets/Cornell-University/arxiv):
- 2.4+ million research papers
- Metadata includes: title, abstract, categories, authors
- JSON format (one paper per line)
- Updated regularly by arXiv

**Note**: You can download this dataset from Kaggle or use your own paper collection!

## 🔐 Security Features

- Password hashing using Werkzeug
- Email verification before account activation
- Session management for authenticated users
- Admin-only routes with authorization decorators
- SQL injection protection via SQLAlchemy ORM
- CSRF protection (implement Flask-WTF for production)

## 🚀 Deployment

### Production Recommendations

1. **Change Secret Key**
```python
app.config['SECRET_KEY'] = 'your-secure-random-key-here'
```

2. **Use Production Database**
```python
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://user:pass@host/db'
```

3. **Set Debug to False**
```python
app.run(debug=False)
```

4. **Use Production WSGI Server**
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

5. **Enable HTTPS** with SSL certificates

6. **Implement Rate Limiting** for API endpoints

## 👥 Team

### Research Team
- **Seetharama Kartheek** - [LinkedIn](https://www.linkedin.com/in/srkch/)
- **Shaik Sameed** - [LinkedIn](https://www.linkedin.com/in/sk-sameed-0909s)

### Design Team
- **K. Sri Akshaya** - [LinkedIn](https://www.linkedin.com/in/sri-akshaya-kodali-619279323/)
- **P V Sai Tejaswi** - [LinkedIn](https://www.linkedin.com/in/tejaswi-putluri-815228291/)

### Engineering Team
- **Karani Sumana Sree** - [LinkedIn](https://www.linkedin.com/in/sumana-sree-karani-13a888300)

## 📧 Contact

For questions or feedback, reach out to:
- Email: seetharamakartheekchallapalli@gmail.com

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- arXiv.org for providing research paper dataset
- CatBoost team for the excellent ML framework
- Flask community for comprehensive documentation
- All contributors and team members

## 🐛 Known Issues

- Email verification requires valid Gmail SMTP credentials
- Large abstracts may take longer to process
- Mobile UI can be improved for admin dashboard

## 🔮 Future Enhancements

- [ ] Support for more academic categories
- [ ] PDF upload and automatic text extraction
- [ ] Batch classification for multiple papers
- [ ] API endpoint for programmatic access
- [ ] Enhanced analytics and visualization
- [ ] Export classification history to CSV
- [ ] Multi-language support
- [ ] Integration with arXiv API
- [ ] Real-time model retraining with user feedback
- [ ] A/B testing for model improvements
- [ ] Docker containerization for easy deployment

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### Ways to Contribute

1. **Improve the Model**
   - Add more training data
   - Experiment with different ML algorithms
   - Optimize hyperparameters
   - Share your trained models

2. **Enhance Features**
   - Add new categories
   - Improve UI/UX
   - Add data visualizations
   - Implement new features

3. **Fix Bugs**
   - Report issues
   - Submit bug fixes
   - Improve error handling

4. **Documentation**
   - Improve README
   - Add code comments
   - Create tutorials
   - Translate documentation

### Contribution Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Model Training Contributions

If you've trained a better model:
1. Share your training notebook with improvements
2. Document your approach and parameters
3. Include performance metrics comparison
4. Submit model files with detailed description

## 📈 Model Performance

### Current Model Metrics

- **Overall Accuracy**: 87.56%
- **Training Data**: 15,000 arXiv papers (3,000 per category)
- **Architecture**: Hybrid CatBoost with TF-IDF (15,000 features) + Domain Keywords (50+ per category)
- **Training Time**: 6-8 minutes on Google Colab
- **Model Parameters**:
  - Iterations: 400
  - Learning Rate: 0.15
  - Tree Depth: 6
  - Early Stopping: 50 rounds

### Feature Engineering

**Hybrid Approach:**
1. **TF-IDF Features** (15,000 dimensions)
   - Bigrams (1-2 word combinations)
   - Sublinear TF scaling
   - Min document frequency: 2
   - Max document frequency: 95%

2. **Domain-Specific Keywords** (5 dimensions)
   - 50+ keywords per category
   - Captures field-specific terminology
   - Examples: "neural network" (AI), "quantum entanglement" (Physics)

### Performance by Category

| Category | Training Papers | Keywords | Estimated Accuracy |
|----------|----------------|----------|-------------------|
| AI & ML | 3,000 | 16 keywords | ~89% |
| Physics | 3,000 | 14 keywords | ~85% |
| Mathematics | 3,000 | 14 keywords | ~88% |
| Biology & Health | 3,000 | 14 keywords | ~87% |
| Chemistry & Materials | 3,000 | 10 keywords | ~89% |

*Note: Run the training notebook to get exact metrics for your trained model*

### Improving Model Accuracy

Want better results? Try these approaches:

**1. Increase Training Data**
```python
target_per_class = 5000  # From 3000 → better generalization
```

**2. More Iterations**
```python
iterations=800  # From 400 → more learning (takes longer)
```

**3. Deeper Trees**
```python
depth=8  # From 6 → capture more complex patterns
```

**4. Add More Keywords**
```python
# Add domain-specific terms you notice in misclassifications
DOMAIN_KEYWORDS['AI_ML'].extend(['bert', 'gpt', 'llm', 'embedding'])
```

**5. Tune TF-IDF**
```python
max_features=20000  # From 15000 → richer vocabulary
ngram_range=(1,3)   # From (1,2) → include trigrams
```

### Training Data Breakdown

**Total Dataset**: 15,000 papers
- Training: 10,500 papers (70%)
- Validation: 2,250 papers (15%)
- Test: 2,250 papers (15%)

**Category Distribution** (balanced):
- AI & ML: 3,000 papers
- Physics: 3,000 papers
- Mathematics: 3,000 papers
- Biology & Health: 3,000 papers
- Chemistry & Materials: 3,000 papers

### Text Preprocessing

The model uses aggressive cleaning:
```python
def clean_text(text):
    - Remove LaTeX math: $equation$
    - Remove LaTeX commands: \command
    - Keep only letters and spaces
    - Convert to lowercase
    - Remove extra whitespace
```

This ensures robust classification even with complex academic notation!

---

**Made with ❤️ by the arXiv Classifier Team**
