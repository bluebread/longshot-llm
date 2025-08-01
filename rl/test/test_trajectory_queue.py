import pytest
import threading
import queue
from lsutils import TrajectoryQueueAgent, TrajectoryMessage

host = 'rabbitmq-bread'
port = 5672


@pytest.fixture(scope="function")
def que():
    que = TrajectoryQueueAgent(host, port)
    yield que
    que.close()

def test_pingpong(que: TrajectoryQueueAgent):
    def time_is_up():
        raise TimeoutError("Time's up!")

    msg = {
        "num_vars": 3,
        "width": 2,
        "base_size": 5,
        "timestamp": "2025-07-21T12:00:00Z",
        "trajectory": {
            "base_formula_id": "f123",
            "steps": [
                {
                    "order": 0,
                    "token_type": "ADD",
                    "token_literals": 5343,
                    "reward": 0.1,
                    "avgQ": 2.5
                },
                {
                    "order": 1,
                    "token_type": "DEL",
                    "token_literals": 616,
                    "reward": -0.05,
                    "avgQ": 3.0
                }
            ]
        }
    }
    
    msg = TrajectoryMessage(**msg)
    que.push(msg)
    timer = threading.Timer(10, time_is_up)
    timer.start()
    
    while (ans := que.pop()) == None:
        pass
    
    timer.cancel()
    assert ans == msg, "The popped message does not match the pushed message."
    
def test_consuming_multiple(que: TrajectoryQueueAgent):
    msg_count = 10
    result_queue = queue.Queue()

    def task():
        th_que = TrajectoryQueueAgent(host, port)
        
        def callback(message):
            result_queue.put(message)
            # Stop after collecting all messages
            if result_queue.qsize() == msg_count:
                th_que.connection.add_callback_threadsafe(th_que.channel.stop_consuming)
                
        th_que.start_consuming(callback)

    def time_is_up():
        raise TimeoutError("Time's up!")
    
    # Clear any previous messages in the queue to ensure clean test
    que.channel.queue_purge(que.queue_name)

    # Start the consumer thread
    consumer_thread = threading.Thread(target=task)
    consumer_thread.start()

    # Push 10 test messages
    for i in range(msg_count):
        msg = TrajectoryMessage(
            num_vars=i,
            width=2,
            base_size=5,
            timestamp="2025-07-21T12:00:00Z",
            trajectory={
                "base_formula_id": f"f{i}",
                "steps": []
            }
        )
        que.push(msg)

    # # Wait for consumer thread to finish (with a timeout safety)
    consumer_thread.join(timeout=10)

    # # Verify all messages were received
    received = []
    while not result_queue.empty():
        received.append(result_queue.get())

    assert len(received) == msg_count, f"Expected {msg_count} messages, got {len(received)}"
    received = sorted([msg['num_vars'] for msg in received])
    expected = list(range(msg_count))
    assert received == expected, "Messages mismatch or out of order"
    
if __name__ == "__main__":
    pytest.main([__file__])