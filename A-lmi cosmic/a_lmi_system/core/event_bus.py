"""
Event Bus: Apache Kafka-based microkernel communication layer
Provides decoupled, asynchronous messaging between A-LMI components
"""

import json
import logging
from typing import Dict, Any, Callable, Optional
from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import KafkaError
import threading

logger = logging.getLogger(__name__)


class EventBus:
    """
    Event-driven microkernel for A-LMI system.
    
    All components (Managers) communicate asynchronously via this EventBus,
    ensuring decoupling, scalability, and resilience.
    """
    
    def __init__(self, bootstrap_servers: list[str], topics: Dict[str, str]):
        """
        Initialize the EventBus.
        
        Args:
            bootstrap_servers: List of Kafka broker addresses
            topics: Dictionary mapping logical topic names to Kafka topic names
        """
        self.bootstrap_servers = bootstrap_servers
        self.topics = topics
        self.producer: Optional[KafkaProducer] = None
        self.consumers: Dict[str, KafkaConsumer] = {}
        self.subscribers: Dict[str, list[Callable]] = {}
        self._lock = threading.Lock()
        
    def connect(self):
        """Establish connection to Kafka."""
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                retries=5
            )
            logger.info(f"EventBus connected to {self.bootstrap_servers}")
        except KafkaError as e:
            logger.error(f"Failed to connect to Kafka: {e}")
            raise
            
    def disconnect(self):
        """Close all connections."""
        if self.producer:
            self.producer.close()
            
        for consumer in self.consumers.values():
            consumer.close()
            
        logger.info("EventBus disconnected")
    
    def publish(self, topic_key: str, event: Dict[str, Any], key: Optional[str] = None):
        """
        Publish an event to the event bus.
        
        Args:
            topic_key: Logical topic name (from config)
            event: Event data as dictionary
            key: Optional message key for partitioning
        """
        if self.producer is None:
            raise RuntimeError("EventBus not connected")
            
        topic_name = self.topics.get(topic_key)
        if not topic_name:
            raise ValueError(f"Unknown topic: {topic_key}")
        
        try:
            future = self.producer.send(topic_name, value=event, key=key)
            # Block for async send with timeout
            record_metadata = future.get(timeout=10)
            logger.debug(f"Published to {topic_name}: {event.get('event_type', 'unknown')}")
        except KafkaError as e:
            logger.error(f"Failed to publish to {topic_name}: {e}")
            raise
    
    def subscribe(self, topic_key: str, callback: Callable[[Dict[str, Any]], None]):
        """
        Subscribe to events from a topic.
        
        Args:
            topic_key: Logical topic name (from config)
            callback: Function to call when event is received
        """
        topic_name = self.topics.get(topic_key)
        if not topic_name:
            raise ValueError(f"Unknown topic: {topic_key}")
        
        with self._lock:
            if topic_key not in self.subscribers:
                self.subscribers[topic_key] = []
                
                # Start consumer thread for this topic
                consumer = KafkaConsumer(
                    topic_name,
                    bootstrap_servers=self.bootstrap_servers,
                    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                    auto_offset_reset='latest',
                    enable_auto_commit=True
                )
                
                self.consumers[topic_key] = consumer
                
                # Start listener thread
                thread = threading.Thread(
                    target=self._listen,
                    args=(topic_key, callback),
                    daemon=True
                )
                thread.start()
            else:
                self.subscribers[topic_key].append(callback)
        
        logger.info(f"Subscribed to {topic_key}")
    
    def _listen(self, topic_key: str, callback: Callable[[Dict[str, Any]], None]):
        """
        Internal method to listen for events on a topic.
        
        Args:
            topic_key: Logical topic name
            callback: Callback function to invoke
        """
        consumer = self.consumers[topic_key]
        
        try:
            for message in consumer:
                try:
                    event = message.value
                    logger.debug(f"Received from {topic_key}: {event.get('event_type', 'unknown')}")
                    callback(event)
                except Exception as e:
                    logger.error(f"Error processing event from {topic_key}: {e}")
        except KeyboardInterrupt:
            logger.info(f"Stopping listener for {topic_key}")
        except Exception as e:
            logger.error(f"Listener error for {topic_key}: {e}")

