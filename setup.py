from setuptools import setup, find_packages

setup(
    name="InsightPulseAI",
    version="1.0.0",
    author="Osman Hassan Abdi",
    author_email="Oabdi44@gmail.com",
    description="Advanced NLP Review Analyzer with Modern AI Techniques",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "streamlit>=1.28.0",
        "pandas>=1.5.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "transformers>=4.30.0",
        "torch>=2.0.0",
        "textblob>=0.17.1",
        "plotly>=5.15.0",
        "wordcloud>=1.9.2",
        "nltk>=3.8",
        "vaderSentiment>=3.3.2",
        "sentence-transformers>=2.2.2",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
        ]
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)