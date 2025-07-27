import json
import pika

class TrajectoryQueue:
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

    def push(self, trajectory: dict) -> None:
        """
        Pushes a trajectory to the RabbitMQ queue.

        :param trajectory: The trajectory to be pushed.
        """
        if not isinstance(trajectory, dict):
            raise ValueError("Trajectory must be a dictionary.")
        
        # TODO: Validate trajectory structure by Pydantic model 
        
        message = json.dumps(trajectory)  # Convert dict to JSON string
        self.channel.basic_publish(
            exchange=self.exchange_name,
            routing_key=self.routing_key,
            body=message,
            properties=pika.BasicProperties(
                content_type='application/json',
                delivery_mode=2,     # Make message persistent 
            ) 
        )

    def pop(self) -> dict | None:
        """
        Pops a trajectory from the RabbitMQ queue.

        :return: The popped trajectory or None if the queue is empty.
        """
        method_frame, _, body = self.channel.basic_get(self.queue_name)
        if method_frame:
            self.channel.basic_ack(method_frame.delivery_tag)
            return json.loads(body)  # Convert string back to dict
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