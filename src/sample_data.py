"""
Preloaded Sample Resumes and Job Descriptions for 1-Click Interactive Evaluation
"""

SAMPLE_RESUMES = {
    "Senior Machine Learning Engineer (Alex Rivera)": """
Alex Rivera
San Francisco, CA | alex.rivera@email.com | (555) 382-9102 | linkedin.com/in/alexrivera-ai | github.com/alexrivera-ai

PROFESSIONAL SUMMARY
Senior Machine Learning Engineer with 6+ years of experience designing, training, and deploying large-scale deep learning models and distributed ML pipelines. Specialized in NLP, Transformers, LLM fine-tuning, and low-latency inference serving with TensorRT and Triton.

TECHNICAL SKILLS
• Programming Languages: Python, C++, SQL, Bash
• Machine Learning & AI: PyTorch, TensorFlow, Scikit-learn, Hugging Face, Transformers, Deep Learning, NLP, RAG, LangChain, MLOps, MLflow, Weights & Biases
• Big Data & Storage: Apache Spark, Kafka, PostgreSQL, Redis, Qdrant, Pinecone, FAISS
• Cloud & DevOps: AWS (EC2, S3, SageMaker), Docker, Kubernetes, Terraform, CI/CD, GitHub Actions

PROFESSIONAL EXPERIENCE
Senior Machine Learning Engineer | NeuralScale AI | 2022 - Present
• Architected and deployed an enterprise Retrieval-Augmented Generation (RAG) platform using PyTorch, Qdrant, and Hugging Face, serving 2.5M daily active users.
• Optimized transformer model inference latency by 42% (from 180ms to 104ms) utilizing TensorRT and Triton Inference Server on NVIDIA A100 GPUs.
• Spearheaded migration of offline training pipelines to Apache Spark and Ray, slashing training data processing time by 65% and saving $140,000 annually in AWS compute costs.
• Mentored 5 junior machine learning engineers and instituted rigorous MLOps practices with MLflow and automated regression benchmarking.

Machine Learning Engineer | DataVision Technologies | 2019 - 2022
• Engineered automated text classification and entity extraction models using BERT and Scikit-learn, boosting F1-score from 0.81 to 0.93.
• Implemented continuous retraining pipelines in Docker and Kubernetes, reducing model deployment cycle time from 3 weeks to 4 hours.
• Reduced false-positive prediction rates by 28% through custom feature engineering and threshold calibration across 10M+ transaction records.

EDUCATION
Master of Science in Computer Science (Machine Learning Specialization)
Stanford University | 2017 - 2019 | GPA: 3.9/4.0

Bachelor of Science in Electrical Engineering & Computer Science
UC Berkeley | 2013 - 2017

KEY PROJECTS
• NeuroSearch: Open-source semantic code search engine utilizing dense bi-encoder embeddings and vector search (2,400+ GitHub stars).
• AutoQuant: Lightweight library for dynamic 4-bit and 8-bit post-training quantization of LLMs in PyTorch.
""",

    "Full-Stack / Backend Engineer (Sarah Chen)": """
Sarah Chen
Seattle, WA | sarah.chen.dev@email.com | (555) 749-1830 | linkedin.com/in/sarahchen-dev | github.com/sarahchen-code

PROFESSIONAL SUMMARY
Results-driven Full-Stack Software Engineer with 5+ years of experience architecting high-throughput distributed backend services and responsive modern web applications. Expert in Python, FastAPI, Node.js, React, TypeScript, and cloud-native databases.

TECHNICAL SKILLS
• Languages: Python, TypeScript, JavaScript, Go, SQL, HTML5, CSS3
• Frameworks & Web: FastAPI, Django, React, Next.js, Node.js, Express, Tailwind CSS, GraphQL, REST API
• Databases: PostgreSQL, MongoDB, Redis, Elasticsearch
• Cloud & Tools: AWS, Docker, Kubernetes, Linux, Git, CI/CD, Jest, Pytest

PROFESSIONAL EXPERIENCE
Senior Backend Engineer | CloudWave Systems | 2021 - Present
• Architected core asynchronous payment and subscription microservices in Python and FastAPI, handling 15,000+ requests per second with 99.99% uptime.
• Refactored legacy monolithic PostgreSQL queries and implemented Redis caching layers, cutting median API response latency by 55%.
• Deployed event-driven microservices using Kafka and Docker on AWS ECS, streamlining asynchronous message processing across 8 internal services.
• Orchestrated automated end-to-end testing with Pytest and GitHub Actions, boosting test coverage from 62% to 94%.

Full-Stack Developer | InnovateTech Labs | 2019 - 2021
• Developed responsive SaaS analytics dashboard using React, Next.js, TypeScript, and Tailwind CSS, increasing user engagement metrics by 35%.
• Implemented secure JWT authentication and role-based access control (RBAC) across 20+ RESTful API endpoints.
• Automated database migration scripts and schema indexing, decreasing daily backup windows by 40%.

EDUCATION
Bachelor of Science in Software Engineering
University of Washington | 2015 - 2019

PROJECTS
• FastCache: High-performance in-memory key-value cache written in Go with LRU eviction and REST interface.
• DevPortal: Developer documentation and API playground built with React and Next.js.
""",

    "Staff DevOps & Cloud Architect (Marcus Vance)": """
Marcus Vance
Austin, TX | marcus.vance.cloud@email.com | (555) 492-3810 | linkedin.com/in/marcusvance-devops | github.com/marcusvance-ops

PROFESSIONAL SUMMARY
Principal Cloud Infrastructure & Site Reliability Engineer with 8+ years designing zero-downtime multi-region Kubernetes clusters on AWS and GCP. Expert in Infrastructure as Code (Terraform), CI/CD automation, service mesh, and Datadog observability.

TECHNICAL SKILLS
• Cloud & Infrastructure: AWS, GCP, Docker, Kubernetes, Helm, Terraform, Ansible, Linux, Bash
• Observability & Security: Prometheus, Grafana, Datadog, ELK Stack, Vault, IAM, OAuth2
• Networking & Protocols: Nginx, Envoy, gRPC, VPC, DNS, Cloudflare, WebSockets
• CI/CD & Automation: GitHub Actions, GitLab CI, Jenkins, ArgoCD, Python, Go

PROFESSIONAL EXPERIENCE
Staff Cloud Architect | CloudNative Dynamics | 2021 - Present
• Designed and executed multi-region AWS EKS Kubernetes migration across 4 global regions, achieving 99.995% service availability.
• Engineered automated Terraform IaC modules managing 500+ cloud resources, eliminating manual provisioning errors and slashing provisioning time from 2 days to 15 minutes.
• Reduced enterprise AWS monthly infrastructure spend by 32% ($280,000 annual savings) via EC2 Spot instance automation and dynamic pod autoscaling (KEDA).
• Led incident response team and decreased mean-time-to-resolution (MTTR) by 48% through automated Prometheus alert thresholds and Datadog APM tracing.

DevOps Engineer | Apex Distributed Systems | 2018 - 2021
• Built enterprise CI/CD pipelines in GitHub Actions and ArgoCD, accelerating production release velocity by 400% (from bi-weekly to 15 deployments daily).
• Hardened Linux container security and automated vulnerability scans with Trivy and SonarQube, resolving 150+ critical CVEs.

EDUCATION
Bachelor of Science in Computer Engineering
University of Texas at Austin | 2014 - 2018
""",

    "Lead Data & Analytics Engineer (Priya Sharma)": """
Priya Sharma
New York, NY | priya.sharma.data@email.com | (555) 612-8923 | linkedin.com/in/priyasharma-data | github.com/priyasharma-analytics

PROFESSIONAL SUMMARY
Senior Data Engineer with 6+ years of expertise architecting high-volume real-time streaming and batch ETL/ELT data pipelines using Apache Spark, Kafka, Snowflake, and dbt. Proven record delivering unified data platforms for enterprise BI and machine learning.

TECHNICAL SKILLS
• Data Engineering: Apache Spark, PySpark, Apache Kafka, Airflow, dbt, Hadoop, Flink
• Databases & Warehousing: Snowflake, BigQuery, PostgreSQL, Databricks, Redshift, ClickHouse
• Languages & Analytics: Python, SQL, Scala, Pandas, Polars, Tableau, Power BI
• Cloud: AWS (S3, EMR, Athena), Docker, Git, CI/CD

PROFESSIONAL EXPERIENCE
Lead Data Engineer | FinMetrics Global | 2022 - Present
• Built streaming financial transaction ingestion pipeline processing 45M events daily using Kafka, PySpark, and Snowflake with sub-second latency.
• Spearheaded migration from legacy Hadoop warehouse to Snowflake and dbt, reducing query execution costs by 52% and cutting morning report latency by 3 hours.
• Implemented automated data quality checks with Great Expectations and Airflow, preventing faulty data loads across 35 downstream analytics tables.

Senior Data Analyst & Engineer | MetroAnalytics | 2018 - 2022
• Developed Tableau executive dashboards and automated SQL models utilized by C-suite leaders to track $50M in quarterly revenue streams.
• Optimized complex dimensional data models and partition pruning in BigQuery, speeding up recurring dashboard load times by 70%.

EDUCATION
Master of Science in Information Systems (Data Analytics)
Columbia University | 2016 - 2018

Bachelor of Technology in Computer Science
IIT Delhi | 2012 - 2016
""",

    "Junior Web Developer (Jordan Lee)": """
Jordan Lee
Chicago, IL | jordan.lee@email.com | (555) 234-5678

SUMMARY
Recent computer science graduate eager to learn frontend and web development. Passionate about coding and modern websites.

SKILLS
HTML, CSS, JavaScript, React basics, Git

EXPERIENCE
Junior Web Intern | Local Digital Agency | 2023 - 2024
• Helped build landing pages for clients using HTML and CSS.
• Worked with team to update blog posts and fix frontend styling bugs.
• Learned React and created small components for internal dashboard.

EDUCATION
Bachelor of Science in Computer Science
University of Illinois Chicago | 2019 - 2023
"""
}

SAMPLE_JOB_DESCRIPTIONS = {
    "Staff / Senior Machine Learning Engineer": """
About the Role:
We are seeking a Senior / Staff Machine Learning Engineer to lead our Core AI and GenAI team. You will be responsible for architecting and deploying large-scale machine learning systems, optimizing transformer architectures, and building production-grade RAG and search infrastructure.

Requirements & Qualifications:
• 5+ years of professional experience in Machine Learning and Deep Learning.
• Strong proficiency in Python, PyTorch, and Scikit-learn.
• Proven track record with Transformers, Hugging Face, LLMs, and RAG architectures.
• Hands-on experience with Vector Databases (Qdrant, Pinecone, or Milvus) and embeddings.
• Production experience with MLOps, MLflow, Docker, Kubernetes, and AWS.
• Deep understanding of model optimization, quantization, TensorRT, or Triton inference serving.
• Strong computer science fundamentals, system design, and communication skills.
""",

    "Senior Backend & Cloud Architect": """
About the Role:
We are looking for a Senior Backend Engineer to design and scale our cloud-native distributed backend services. You will build high-throughput microservices, optimize distributed databases, and own cloud infrastructure on AWS.

Requirements & Qualifications:
• 4+ years of backend development experience using Python, FastAPI, Go, or Java.
• Strong expertise in relational and NoSQL databases: PostgreSQL, Redis, MongoDB.
• Deep experience designing RESTful APIs, gRPC, and event-driven architectures with Kafka or RabbitMQ.
• Solid background in Cloud Infrastructure (AWS), Docker containerization, Kubernetes, and CI/CD pipelines.
• Focus on high availability, microservices scalability, caching strategies, and system design.
• Passion for clean code, unit testing (Pytest), and engineering mentorship.
"""
}
