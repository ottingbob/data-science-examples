from diagrams import Cluster, Diagram, Edge
from diagrams.k8s.clusterconfig import HPA
from diagrams.k8s.compute import Deployment, Pod, ReplicaSet, Job
from diagrams.k8s.network import Ingress, Service
from diagrams.onprem.database import PostgreSQL
from diagrams.onprem.monitoring import Grafana, Prometheus


with Diagram("ML Deployment Pipeline", show=True):
    with Cluster("Monitoring Setup"):
        metrics = Prometheus("metrics")
        metrics << Edge(color="firebrick", style="dashed") << Grafana("monitoring")

    with Cluster("Data Pipeline"):
        pg_db = PostgreSQL("data-warehouse")
        data_pipline = pg_db << Job("data-cruncher") << metrics
    with Cluster("Internal Data API"):
        data_api_svc = Service("internal-data-api")
        internal_data_api = data_api_svc >> [
            Pod("pod1"),
            Pod("pod2"),
            Pod("pod3"),
        ]
        data_api_deploy = Deployment("data-api-deploy")
        (internal_data_api << ReplicaSet("data-api-rs") << data_api_deploy << pg_db)
        data_api_deploy << metrics
    with Cluster("ML Application"):
        net = Ingress("ml.domain.com") >> Service("ml-svc")
        # << Deployment("ml-deployment") << HPA("hpa")
        apps = [
            Pod("pod1"),
            Pod("pod2"),
            Pod("pod3"),
        ]

        deployment = Deployment("ml-deployment")
        (
            net
            >> Edge(color="darkred")
            >> apps
            << Edge(color="darkorange")
            << ReplicaSet("ml-rs")
            << deployment
            << HPA("hpa")
        )
        deployment << metrics
    deployment << data_api_svc
