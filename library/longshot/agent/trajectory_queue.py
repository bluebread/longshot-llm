import json
import pika
from pydantic import BaseModel, Field
from typing import Dict, Any
from datetime import datetime

class TrajectoryStep(BaseModel):
    """
    Model representing a single step in a trajectory.
    It includes the arm ID and the reward received for that step.
    """

    order: int = Field(..., description="Order of the step in the trajectory")
    token_type: str = Field(..., description="Type of the token (e.g., 'arm')")
    token_literals: int = Field(..., description="Literal associated with the token")
    reward: float = Field(..., description="Reward received for this step")
    avgQ: float = Field(..., description="Average Q-value for this step")

class Trajectory(BaseModel):
    """
    Model representing a trajectory in the context of reinforcement learning.
    It includes the trajectory ID, the arm ID, and the trajectory data.
    """
    
    base_formula_id: str | None = Field(None, description="ID of the base formula for the trajectory")
    steps: list[TrajectoryStep] = Field(..., description="List of steps in the trajectory")

class TrajectoryMessage(BaseModel):
    """
    Model representing a trajectory message.
    It includes the trajectory ID, the arm ID, and the trajectory data.
    """
    
    num_vars: int = Field(..., description="Number of variables in the trajectory")
    width: int = Field(..., description="Width of the trajectory")
    base_size: int = Field(..., description="Size of the base formula")
    timestamp: datetime = Field(..., description="Timestamp of the trajectory")
    trajectory: Trajectory = Field(..., description="The trajectory data itself")


class TrajectoryQueueAgent:
    """
    A class to manage a queue of trajectories using RabbitMQ.
    """

    def __init__(self, host: str, port: int = 5672):
        """
        Initializes the TrajectoryQueue with the specified queue name and RabbitMQ host.

        :param queue_name: The name of the RabbitMQ queue.
        :param host: The RabbitMQ server host.
        """
        self.queue_name = 'trajectory.queue'
        self.exchange_name = 'trajectory.exchange'
        self.routing_key = 'trajectory.routing'
        username = 'haowei'
        password = 'bread861122'
        credentials = pika.PlainCredentials(username=username, password=password)
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(host, port, credentials=credentials))
        self.channel = self.connection.channel()
        
        # Declare the exchange and queue
        self.channel.exchange_declare(
            exchange=self.exchange_name, 
            exchange_type='direct', 
            durable=True,
        )
        self.channel.queue_declare(
            queue=self.queue_name, 
            auto_delete=False,
            durable=True,
        )
        self.channel.queue_bind(
            exchange=self.exchange_name,
            queue=self.queue_name,
            routing_key=self.routing_key
        )

    def push(self, trajectory: TrajectoryMessage) -> None:
        """
        Pushes a trajectory to the RabbitMQ queue.

        :param trajectory: The trajectory to be pushed.
        """
        message = trajectory.model_dump_json()  # Convert dict to JSON string
        self.channel.basic_publish(
            exchange=self.exchange_name,
            routing_key=self.routing_key,
            body=message,
            properties=pika.BasicProperties(
                content_type='application/json',
                delivery_mode=2,     # Make message persistent 
            ) 
        )

    def pop(self) -> TrajectoryMessage | None:
        """
        Pops a trajectory from the RabbitMQ queue.

        :return: The popped trajectory or None if the queue is empty.
        """
        method_frame, _, body = self.channel.basic_get(self.queue_name)
        if method_frame:
            self.channel.basic_ack(method_frame.delivery_tag)
            return TrajectoryMessage(**json.loads(body))  # Convert string back to dict
        return None

    def start_consuming(self, callback: callable):
        """
        Starts consuming messages from the RabbitMQ queue.

        :param callback: A function to call with each message.
        """
        def on_message(channel, method, properties, body):
            data = json.loads(body)
            callback(data)
            channel.basic_ack(method.delivery_tag)

        self.channel.basic_consume(
            queue=self.queue_name,
            on_message_callback=on_message,
            auto_ack=False
        )
        self.channel.start_consuming()
        
    def close(self):
        """
        Closes the connection to the RabbitMQ server.
        """
        self.connection.close()