import json
import asyncio
from typing import List, Dict, Any
from datetime import datetime
import pika
import aio_pika
from aio_pika import Message
from pydantic import BaseModel, Field
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
        message = json.dumps(trajectory.model_dump())  # Convert dict to JSON string
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


class AsyncTrajectoryQueueAgent:
    """
    The `AsyncTrajectoryQueueAgent` class provides an asynchronous high-level interface 
    for managing trajectory data using RabbitMQ. It handles the connection setup, 
    batch message publishing, and consumption for trajectory processing in the RL system.
    
    This async version is more efficient for bulk operations and integrates better
    with FastAPI async endpoints.
    
    Usage:
    ------
    
    Basic usage with automatic connection:
    
        # Create and connect in one step
        agent = await AsyncTrajectoryQueueAgent.create("localhost", port=5672)
        
        # Create with custom QoS prefetch count for high-throughput scenarios
        agent = await AsyncTrajectoryQueueAgent.create("localhost", qos_prefetch_count=500)
        
        # Push a single trajectory
        await agent.push(trajectory_message)
        
        # Push multiple trajectories efficiently
        await agent.push_batch(trajectory_list)
        
        # Pop a trajectory (returns None if queue is empty)
        trajectory = await agent.pop()
        
        # Clean up connection
        await agent.close()
    
    Context manager usage (recommended):
    
        async with AsyncTrajectoryQueueAgent("localhost", port=5672) as agent:
            await agent.push(trajectory_message)
            # Connection automatically established and closed
    
    Consumer usage:
    
        agent = await AsyncTrajectoryQueueAgent.create("localhost")
        
        async def process_trajectory(data):
            print(f"Received trajectory: {data}")
        
        await agent.start_consuming(process_trajectory)  # Blocks until stopped
        await agent.close()
    
    Important Notes:
    ---------------
    - Always use the async factory method `create()` instead of direct instantiation
    - Use context manager pattern when possible for automatic cleanup
    - All methods assume an active connection and will fail if not connected
    """

    def __init__(self, host: str, port: int = 5672, qos_prefetch_count: int = 100):
        """
        Initializes the AsyncTrajectoryQueueAgent with the specified RabbitMQ host and port.
        
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
        :param qos_prefetch_count: Number of unacknowledged messages that can be processed simultaneously (default: 100)
        :type qos_prefetch_count: int
        """
        self.host = host
        self.port = port
        self.qos_prefetch_count = qos_prefetch_count
        self.queue_name = 'trajectory.queue'
        self.exchange_name = 'trajectory.exchange'
        self.routing_key = 'trajectory.routing'
        self.username = 'haowei'
        self.password = 'bread861122'
        
        self.connection = None
        self.channel = None
        self.exchange = None
        self.queue = None

    async def connect(self):
        """
        Establishes connection to RabbitMQ server and sets up exchange and queue.
        This method must be called before using push/pop operations.
        """
        # Create connection
        self.connection = await aio_pika.connect_robust(
            f"amqp://{self.username}:{self.password}@{self.host}:{self.port}/"
        )
        
        # Create channel
        self.channel = await self.connection.channel()
        
        # Declare exchange and queue concurrently (they're independent)
        self.exchange, self.queue = await asyncio.gather(
            self.channel.declare_exchange(
                self.exchange_name,
                aio_pika.ExchangeType.DIRECT,
                durable=True
            ),
            self.channel.declare_queue(
                self.queue_name,
                durable=True,
                auto_delete=False
            )
        )
        
        # Bind queue to exchange (depends on both being created)
        await self.queue.bind(self.exchange, self.routing_key)

    @classmethod
    async def create(cls, host: str, port: int = 5672, qos_prefetch_count: int = 100):
        """
        Factory method that creates and connects an AsyncTrajectoryQueueAgent.
        This aligns with the synchronous TrajectoryQueueAgent where connection
        happens immediately during initialization.
        
        :param host: The RabbitMQ server host address
        :type host: str
        :param port: The RabbitMQ server port (default: 5672)
        :type port: int
        :param qos_prefetch_count: Number of unacknowledged messages that can be processed simultaneously (default: 100)
        :type qos_prefetch_count: int
        :return: Connected AsyncTrajectoryQueueAgent instance
        :rtype: AsyncTrajectoryQueueAgent
        """
        instance = cls(host, port, qos_prefetch_count)
        await instance.connect()
        return instance

    async def push(self, trajectory: TrajectoryQueueMessage) -> None:
        """
        Pushes a single trajectory to the RabbitMQ queue asynchronously.
        
        :param trajectory: The trajectory data to be queued
        :type trajectory: TrajectoryQueueMessage
        """
        message_body = json.dumps(trajectory.model_dump())
        message = Message(
            message_body.encode(),
            content_type='application/json',
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT
        )
        
        await self.exchange.publish(
            message,
            routing_key=self.routing_key
        )

    async def push_batch(self, trajectories: List[TrajectoryQueueMessage]) -> int:
        """
        Pushes multiple trajectories to the RabbitMQ queue in batch for better performance.
        
        :param trajectories: List of trajectory data to be queued
        :type trajectories: List[TrajectoryQueueMessage]
        :return: Number of trajectories successfully pushed
        :rtype: int
        """
        if not trajectories:
            return 0
            
        # Use publisher confirms for reliability
        await self.channel.set_qos(prefetch_count=self.qos_prefetch_count)
        
        pushed_count = 0
        
        # Process trajectories in batches to avoid overwhelming the server
        batch_size = 50
        for i in range(0, len(trajectories), batch_size):
            batch = trajectories[i:i + batch_size]
            
            # Create all messages for this batch
            messages = []
            for trajectory in batch:
                message_body = json.dumps(trajectory.model_dump())
                message = Message(
                    message_body.encode(),
                    content_type='application/json',
                    delivery_mode=aio_pika.DeliveryMode.PERSISTENT
                )
                messages.append(message)
            
            # Publish all messages in the batch concurrently
            tasks = [
                self.exchange.publish(msg, routing_key=self.routing_key)
                for msg in messages
            ]
            
            # Wait for all messages in this batch to be published
            await asyncio.gather(*tasks)
            pushed_count += len(batch)
        
        return pushed_count

    async def pop(self) -> TrajectoryQueueMessage | None:
        """
        Pops a trajectory from the RabbitMQ queue asynchronously.
        
        :return: The popped trajectory data or None if queue is empty
        :rtype: TrajectoryQueueMessage | None
        """
        try:
            message = await self.queue.get(timeout=1.0)
            if message:
                async with message.process():
                    data = json.loads(message.body.decode())
                    return TrajectoryQueueMessage(**data)
        except aio_pika.exceptions.QueueEmpty:
            return None
        
        return None

    async def start_consuming(self, callback):
        """
        Starts consuming messages from the RabbitMQ queue asynchronously.
        
        :param callback: Async function to be called with trajectory data
        :type callback: callable
        """
        async def process_message(message: aio_pika.abc.AbstractIncomingMessage):
            async with message.process():
                data = json.loads(message.body.decode())
                await callback(data)
        
        await self.queue.consume(process_message)

    async def close(self):
        """
        Closes the connection to the RabbitMQ server asynchronously.
        """
        if self.connection and not self.connection.is_closed:
            await self.connection.close()
            
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()