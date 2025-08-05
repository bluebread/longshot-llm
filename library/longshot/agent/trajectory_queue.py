import json
import pika
from pydantic import BaseModel, Field
from typing import Dict, Any
from datetime import datetime
from ..models.trajectory import TrajectoryQueueMessage

class TrajectoryQueueAgent:
    """
    The `TrajectoryQueueAgent` class provides a high-level interface for managing trajectory data using RabbitMQ. It handles the connection setup, message publishing, and consumption for trajectory processing in the RL system.
    """

    def __init__(self, host: str, port: int = 5672):
        """
        Initializes the TrajectoryQueueAgent with the specified RabbitMQ host and port. This constructor establishes a connection to the RabbitMQ server, declares the necessary exchange and queue, and sets up the binding between them.
        
        The following RabbitMQ components are automatically configured:
            - Queue Name: `trajectory.queue`
            - Exchange Name: `trajectory.exchange`
            - Exchange Type: `direct`
            - Routing Key: `trajectory.routing`
            - Durability: `true` (both queue and exchange persist across server restarts)
        
        :param host: The RabbitMQ server host address
        :type host: str
        :param port: The RabbitMQ server port (default: 5672)
        :type port: int
        """
        self.queue_name = 'trajectory.queue'
        self.exchange_name = 'trajectory.exchange'
        self.routing_key = 'trajectory.routing'
        username = 'haowei'
        password = 'bread861122'
        credentials = pika.PlainCredentials(username=username, password=password)
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(
                host, 
                port, 
                credentials=credentials
            )
        )
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

    def push(self, trajectory: TrajectoryQueueMessage) -> None:
        """
        Pushes a trajectory to the RabbitMQ queue. The trajectory is serialized to JSON format before being published to the queue.
        
        :param trajectory: The trajectory data to be queued, must be an instance of TrajectoryQueueMessage
        :type trajectory: TrajectoryQueueMessage
        :raises ValueError: If the trajectory parameter is not an instance of TrajectoryQueueMessage
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

    def pop(self) -> TrajectoryQueueMessage | None:
        """
        Pops a trajectory from the RabbitMQ queue using basic_get for immediate retrieval. If a message is available, it is acknowledged and returned as a TrajectoryQueueMessage instance. If the queue is empty, returns None.
        
        :return: The popped trajectory data as a TrajectoryQueueMessage instance or None if the queue is empty
        :rtype: TrajectoryQueueMessage | None
        :raises ValueError: If the message body cannot be converted to TrajectoryQueueMessage
        """
        method_frame, _, body = self.channel.basic_get(self.queue_name)
        if method_frame:
            self.channel.basic_ack(method_frame.delivery_tag)
            return TrajectoryQueueMessage(**json.loads(body))  # Convert string back to dict
        return None

    def start_consuming(self, callback: callable):
        """
        Starts consuming messages from the RabbitMQ queue. This method will block and continuously listen for incoming messages, processing each one using the provided callback function.
        
        :param callback: A function that will be called with the trajectory data as a dictionary when a message is received
        :type callback: callable
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
        Closes the connection to the RabbitMQ server. This method should be called to properly clean up resources when the TrajectoryQueue is no longer needed.
        """
        self.connection.close()