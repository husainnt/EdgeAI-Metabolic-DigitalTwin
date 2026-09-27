// TwinPlaybackController.cs
// ==========================================================
// First-integration stub: reads the JSON exported by
// export_twin_visualization_data.py and plays it back frame-by-frame,
// driving organ color and heart-pulse speed from REAL numbers (glucose,
// pancreas secretion proxy, heart rate) rather than a fixed animation loop.
//
// NOT YET COMPILED/TESTED IN UNITY -- I don't have a Unity environment to
// verify this in. The JSON parsing, coroutine timing, and Mathf/Color API
// usage should be correct, but expect to wire up the Inspector references
// to your actual scene hierarchy together, and fix any small API mismatch
// (e.g. if you're using TextMeshPro instead of legacy UI.Text, swap the
// using/type below).
//
// Setup:
//   1. Attach this script to any GameObject in the scene.
//   2. Drag the exported .json file into the "Json File" slot as a
//      TextAsset (Unity auto-imports .json as TextAsset if it's under
//      Assets/ or a Resources folder).
//   3. Drag your Pancreas mesh's Renderer into "Pancreas Renderer".
//   4. Drag your Heart Transform into "Heart Transform" (for the pulse
//      scale animation).
//   5. Optionally hook up glucoseText / heartRateText (legacy UI.Text) for
//      on-screen readout.
//   6. Press Play.
//
// liver_output_proxy is intentionally NOT wired here -- the Python export
// sends it as null because no real value exists yet (see that script's
// docstring). Don't animate the liver from this data until that's real.
//
// degradation_level in the JSON is DEMO-ONLY (cycles 1-5 for testing your
// ladder visuals, not a real inference output) -- OnDegradationLevelChanged
// below is a hook for you to extend with your own ladder visualization
// once you decide how you want to represent it (e.g. greying out sensor
// icons, showing a wider uncertainty band).

using System.Collections;
using UnityEngine;
using UnityEngine.UI;

public class TwinPlaybackController : MonoBehaviour
{
    [Header("Data source")]
    public TextAsset jsonFile;
    [Tooltip("Seconds of real time per exported frame during playback (each frame = 5 real minutes of patient data)")]
    public float secondsPerFrame = 0.5f;

    [Header("Pancreas — color driven by real secretion proxy")]
    public Renderer pancreasRenderer;
    public Color pancreasHealthyColor = new Color(0.3f, 0.8f, 0.4f); // low secretion / near-basal
    public Color pancreasStressedColor = new Color(0.85f, 0.2f, 0.2f); // near max_secretion
    [Tooltip("From the export's meta.field_notes.pancreas_secretion_proxy range -- update these to match the patient you exported")]
    public float secretionProxyMin = 0.0211f; // basal_rate
    public float secretionProxyMax = 0.05f;   // controller's max_secretion cap

    [Header("Heart — pulse speed driven by real HR")]
    public Transform heartTransform;
    public float heartPulseAmplitude = 0.08f;
    private Vector3 _heartBaseScale;
    private float _currentHR = 70f;

    [Header("Optional on-screen readout (legacy UI.Text -- swap for TMP_Text if that's what your scene uses)")]
    public Text glucoseText;
    public Text heartRateText;

    private TwinExport _data;

    void Start()
    {
        if (jsonFile == null)
        {
            Debug.LogError("TwinPlaybackController: no jsonFile assigned.");
            return;
        }
        _data = JsonUtility.FromJson<TwinExport>(jsonFile.text);
        if (_data == null || _data.frames == null || _data.frames.Length == 0)
        {
            Debug.LogError("TwinPlaybackController: JSON parsed but has no frames -- check the export ran correctly.");
            return;
        }
        Debug.Log($"Loaded twin export for patient {_data.meta.patient}: {_data.meta.n_frames} frames");

        if (heartTransform != null)
            _heartBaseScale = heartTransform.localScale;

        StartCoroutine(PlaybackLoop());
    }

    void Update()
    {
        // Here I drive the heart pulse continuously (not just once per
        // playback frame) so the beat looks smooth between data steps,
        // using whatever the current frame's real HR was
        if (heartTransform != null)
        {
            float beatsPerSecond = _currentHR / 60f;
            float pulse = 1f + heartPulseAmplitude * Mathf.Sin(Time.time * beatsPerSecond * 2f * Mathf.PI);
            heartTransform.localScale = _heartBaseScale * pulse;
        }
    }

    private IEnumerator PlaybackLoop()
    {
        int lastDegradationLevel = -1;

        foreach (TwinFrame frame in _data.frames)
        {
            _currentHR = frame.heart_rate;

            if (pancreasRenderer != null)
            {
                float t = Mathf.InverseLerp(secretionProxyMin, secretionProxyMax, frame.pancreas_secretion_proxy);
                pancreasRenderer.material.color = Color.Lerp(pancreasHealthyColor, pancreasStressedColor, t);
            }

            if (glucoseText != null)
                glucoseText.text = $"Glucose: {frame.glucose_hybrid:F0} mg/dL  (physics-only: {frame.glucose_physics:F0})";
            if (heartRateText != null)
                heartRateText.text = $"HR: {frame.heart_rate:F0} bpm";

            if (frame.degradation_level != lastDegradationLevel)
            {
                OnDegradationLevelChanged(frame.degradation_level);
                lastDegradationLevel = frame.degradation_level;
            }

            yield return new WaitForSeconds(secondsPerFrame);
        }
        Debug.Log("TwinPlaybackController: reached the end of the exported record.");
    }

    // Here I leave this as a hook for you to extend -- decide how the
    // 5-level graceful-degradation ladder should look (dimmed sensor
    // icons, a wider uncertainty band on the glucose readout, etc.).
    // Remember: the levels in THIS demo file are cycling for testing
    // purposes only, not a real inference output.
    private void OnDegradationLevelChanged(int level)
    {
        Debug.Log($"Degradation level changed to {level}");
    }
}

[System.Serializable]
public class TwinFrame
{
    public float glucose_real;
    public float glucose_physics;
    public float glucose_hybrid;
    public float heart_rate;
    public float pancreas_secretion_proxy;
    public int degradation_level;
    public int meal_event_active;
    // liver_output_proxy intentionally not declared here -- it's always
    // null in the JSON right now, and JsonUtility ignores JSON fields with
    // no matching C# field, so this is safe to leave out until it's real.
}

[System.Serializable]
public class TwinMeta
{
    public string patient;
    public string split;
    public int n_frames;
    public int frame_interval_minutes;
    // field_notes is a JSON object-of-strings (a dictionary) -- JsonUtility
    // cannot parse Dictionary<>, so it's intentionally omitted here too.
    // Read that field by opening the JSON file directly if you need the
    // human-readable descriptions/ranges.
}

[System.Serializable]
public class TwinExport
{
    public TwinMeta meta;
    public TwinFrame[] frames;
}