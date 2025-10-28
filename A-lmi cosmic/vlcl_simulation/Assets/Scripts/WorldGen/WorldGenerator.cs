using UnityEngine;

namespace VLCL.WorldGen
{
    /// <summary>
    /// Procedural World Generation
    /// Implements Genesis blueprint with Phi, Chaos Theory, and E=mc^2 influences
    /// </summary>
    public class WorldGenerator : MonoBehaviour
    {
        [Header("Terrain Parameters")]
        public int mapWidth = 512;
        public int mapHeight = 512;
        public float scale = 10f;

        [Header("Mathematical Influences")]
        public float phiInfluence = 1.618f;  // Golden Ratio
        public float chaosInfluence = 0.3f;  // Chaos Theory (Perlin)
        public float eMc2Influence = 0.2f;   // E=mc^2 (exponential)

        private TerrainData terrainData;
        private Terrain terrain;

        void Start()
        {
            GenerateTerrain();
        }

        public void GenerateTerrain()
        {
            terrainData = new TerrainData();
            terrainData.heightmapResolution = mapWidth;
            terrainData.size = new Vector3(100, 10, 100);

            float[,] heights = new float[mapWidth, mapHeight];

            for (int x = 0; x < mapWidth; x++)
            {
                for (int y = 0; y < mapHeight; y++)
                {
                    heights[x, y] = CalculateHeight(x, y);
                }
            }

            terrainData.SetHeights(0, 0, heights);
            terrain = Terrain.CreateTerrainGameObject(terrainData).GetComponent<Terrain>();
            terrain.transform.position = transform.position;
        }

        private float CalculateHeight(int x, int y)
        {
            float xCoord = (float)x / mapWidth * scale;
            float yCoord = (float)y / mapHeight * scale;

            float heightValue = 0f;

            // Biome-specific base noise (Chaos Theory)
            heightValue += chaosInfluence * Mathf.PerlinNoise(xCoord * 0.3f, yCoord * 0.3f);

            // Golden Ratio (Phi) influence (Harmonic)
            heightValue += phiInfluence * 0.1f * Mathf.Sin(xCoord) * Mathf.Cos(yCoord);

            // Chaos Theory influence (Turbulence)
            heightValue += 0.3f * Mathf.PerlinNoise(xCoord, yCoord);

            // Einstein's Mass-Energy Equivalence (Exponential curve)
            heightValue += eMc2Influence * Mathf.Exp(-Mathf.Pow(xCoord - yCoord, 2));

            heightValue = Mathf.InverseLerp(-1.0f, 1.0f, heightValue);
            return heightValue;
        }
    }
}

