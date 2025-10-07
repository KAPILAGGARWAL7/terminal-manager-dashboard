from setuptools import setup, find_packages

setup(
    name="ai-dashboard-backend",
    version="1.0.0",
    description="AI-powered dashboard generation backend",
    packages=find_packages(),
    install_requires=[
        "flask==3.0.0",
        "streamlit==1.28.1",
        "pandas==2.1.3",
        "plotly==5.17.0",
        "requests==2.31.0",
        "openpyxl==3.1.2",
        "python-multipart==0.0.6",
        "flask-cors==4.0.0",
        "python-dotenv==1.0.0"
    ],
    python_requires=">=3.8",
    author="Dashboard Team",
    author_email="team@dashboard.com",
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