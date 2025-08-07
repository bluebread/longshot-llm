
import httpx
import logging
from longshot.agent import TrajectoryQueueAgent
from .processor import TrajectoryProcessor

logging.basicConfig(
    level=logging.INFO, 
    filename="processor.log", 
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="a",
)
logger = logging.getLogger(__name__)

warehouse = httpx.Client(base_url="http://localhost:8000")
trajectory_queue = TrajectoryQueueAgent(host="rabbitmq-bread", port=5672)
trajproc = TrajectoryProcessor(warehouse)

if __name__ == "__main__":
    try:
        def consume_trajectory(data):
            trajproc.process_trajectory(data)

        trajectory_queue.start_consuming(consume_trajectory)
    except KeyboardInterrupt:
        logger.info("Stopping trajectory queue consumption.")
    finally:
        trajectory_queue.close()
        warehouse.close()
        logger.info("Trajectory queue and warehouse connections closed.")
    