import pytest
import pytest_asyncio
import asyncio
import threading
import queue
from datetime import datetime
from longshot.agent import TrajectoryQueueAgent, AsyncTrajectoryQueueAgent
from longshot.models import TrajectoryQueueMessage

host = 'rabbitmq-bread'
port = 5672


class TestTrajectoryQueueAgent:
    """Test suite for synchronous TrajectoryQueueAgent"""
    
    @pytest.fixture(scope="function")
    def que(self):
        que = TrajectoryQueueAgent(host, port)
        yield que
        que.close()

    def test_pingpong(self, que: TrajectoryQueueAgent):
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
        
        msg = TrajectoryQueueMessage(**msg)
        que.push(msg)
        timer = threading.Timer(10, time_is_up)
        timer.start()
        
        while (ans := que.pop()) == None:
            pass
        
        timer.cancel()
        assert ans == msg, "The popped message does not match the pushed message."
        
    def test_consuming_multiple(self, que: TrajectoryQueueAgent):
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
            msg = TrajectoryQueueMessage(
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


class TestAsyncTrajectoryQueueAgent:
    """Test suite for asynchronous AsyncTrajectoryQueueAgent"""
    
    @pytest_asyncio.fixture(scope="function")
    async def async_que(self):
        que = AsyncTrajectoryQueueAgent(host, port)
        await que.connect()
        yield que
        await que.close()

    @pytest.mark.asyncio
    async def test_async_pingpong(self, async_que: AsyncTrajectoryQueueAgent):
        msg = TrajectoryQueueMessage(
            num_vars=3,
            width=2,
            base_size=5,
            timestamp=datetime.fromisoformat("2025-07-21T12:00:00+00:00"),
            trajectory={
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
        )
        
        await async_que.push(msg)
        
        # Try to pop the message with a reasonable timeout
        for _ in range(50):  # 5 second timeout
            ans = await async_que.pop()
            if ans is not None:
                break
            await asyncio.sleep(0.1)
        else:
            pytest.fail("Timeout waiting for message")
        
        assert ans == msg, "The popped message does not match the pushed message."

    @pytest.mark.asyncio
    async def test_async_push_batch(self, async_que: AsyncTrajectoryQueueAgent):
        msg_count = 5
        messages = []
        
        for i in range(msg_count):
            msg = TrajectoryQueueMessage(
                num_vars=i,
                width=2,
                base_size=5,
                timestamp=datetime.fromisoformat("2025-07-21T12:00:00+00:00"),
                trajectory={
                    "base_formula_id": f"f{i}",
                    "steps": []
                }
            )
            messages.append(msg)
        
        # Test batch push
        pushed_count = await async_que.push_batch(messages)
        assert pushed_count == msg_count, f"Expected {msg_count} messages pushed, got {pushed_count}"
        
        # Pop all messages and verify
        received = []
        for _ in range(msg_count * 10):  # Give extra attempts
            msg = await async_que.pop()
            if msg:
                received.append(msg)
            if len(received) == msg_count:
                break
            await asyncio.sleep(0.1)
        
        assert len(received) == msg_count, f"Expected {msg_count} messages, got {len(received)}"
        received_nums = sorted([msg.num_vars for msg in received])
        expected_nums = list(range(msg_count))
        assert received_nums == expected_nums, "Messages mismatch or out of order"

    @pytest.mark.asyncio
    async def test_async_consuming(self, async_que: AsyncTrajectoryQueueAgent):
        msg_count = 3
        received_messages = []
        
        async def callback(data):
            received_messages.append(data)
        
        # Push test messages first
        for i in range(msg_count):
            msg = TrajectoryQueueMessage(
                num_vars=i,
                width=2,
                base_size=5,
                timestamp=datetime.fromisoformat("2025-07-21T12:00:00+00:00"),
                trajectory={
                    "base_formula_id": f"f{i}",
                    "steps": []
                }
            )
            await async_que.push(msg)
        
        # Start consuming in background
        consume_task = asyncio.create_task(async_que.start_consuming(callback))
        
        # Wait for messages to be processed
        for _ in range(100):  # 10 second timeout
            if len(received_messages) >= msg_count:
                break
            await asyncio.sleep(0.1)
        
        # Cancel consuming task
        consume_task.cancel()
        try:
            await consume_task
        except asyncio.CancelledError:
            pass
        
        assert len(received_messages) >= msg_count, f"Expected at least {msg_count} messages, got {len(received_messages)}"

    @pytest.mark.asyncio 
    async def test_async_context_manager(self):
        async with AsyncTrajectoryQueueAgent(host, port) as que:
            msg = TrajectoryQueueMessage(
                num_vars=1,
                width=2,
                base_size=5,
                timestamp=datetime.fromisoformat("2025-07-21T12:00:00+00:00"),
                trajectory={
                    "base_formula_id": "f_context",
                    "steps": []
                }
            )
            await que.push(msg)
            retrieved = await que.pop()
            assert retrieved == msg, "Context manager test failed"

    @pytest.mark.asyncio
    async def test_empty_batch_push(self, async_que: AsyncTrajectoryQueueAgent):
        pushed_count = await async_que.push_batch([])
        assert pushed_count == 0, "Empty batch should return 0"

    @pytest.mark.asyncio
    async def test_pop_empty_queue(self, async_que: AsyncTrajectoryQueueAgent):
        # Clear the queue first
        while await async_que.pop():
            pass
        
        result = await async_que.pop()
        assert result is None, "Popping from empty queue should return None"


if __name__ == "__main__":
    pytest.main([__file__])