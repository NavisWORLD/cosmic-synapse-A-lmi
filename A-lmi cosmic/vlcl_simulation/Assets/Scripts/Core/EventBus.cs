using System.Collections.Generic;
using UnityEngine;

namespace VLCL.Core
{
    /// <summary>
    /// Event-driven microkernel for VLCL simulation.
    /// Topic-based pub/sub messaging system.
    /// </summary>
    public class EventBus : MonoBehaviour
    {
        private static EventBus _instance;
        public static EventBus Instance
        {
            get
            {
                if (_instance == null)
                {
                    _instance = FindObjectOfType<EventBus>();
                    if (_instance == null)
                    {
                        GameObject go = new GameObject("EventBus");
                        _instance = go.AddComponent<EventBus>();
                        DontDestroyOnLoad(go);
                    }
                }
                return _instance;
            }
        }

        private Dictionary<string, System.Action<object>> subscribers = new Dictionary<string, System.Action<object>>();

        private void Awake()
        {
            if (_instance != null && _instance != this)
            {
                Destroy(gameObject);
                return;
            }
            _instance = this;
        }

        public void Publish(string topic, object data)
        {
            if (subscribers.ContainsKey(topic))
            {
                subscribers[topic]?.Invoke(data);
            }
        }

        public void Subscribe(string topic, System.Action<object> callback)
        {
            if (!subscribers.ContainsKey(topic))
            {
                subscribers[topic] = null;
            }
            subscribers[topic] += callback;
        }

        public void Unsubscribe(string topic, System.Action<object> callback)
        {
            if (subscribers.ContainsKey(topic))
            {
                subscribers[topic] -= callback;
            }
        }
    }
}

