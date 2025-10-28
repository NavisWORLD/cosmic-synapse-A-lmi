using UnityEngine;
using UnityEngine.UI;
using VLCL.Core;

namespace VLCL.Managers
{
    /// <summary>
    /// UI Manager: Handles user interface and visualization
    /// </summary>
    public class UIManager : MonoBehaviour
    {
        [Header("UI Elements")]
        public Text statusText;
        public Text audioStatusText;
        public Slider audioSlider;

        private void Start()
        {
            // Subscribe to relevant events
            EventBus.Instance.Subscribe("audio.psd", OnAudioPSD);
        }

        private void OnAudioPSD(object data)
        {
            if (data is float psd)
            {
                audioSlider.value = psd;
                audioStatusText.text = $"Audio PSD: {psd:F2}";
            }
        }

        public void UpdateStatus(string message)
        {
            if (statusText != null)
            {
                statusText.text = message;
            }
        }
    }
}

