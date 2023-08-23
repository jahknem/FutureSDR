// Sonogram fragment shader
#ifdef GL_ES
precision mediump float;
#endif

varying vec2 texCoord;

uniform sampler2D frequencyData;
uniform vec4 foregroundColor;
uniform vec4 backgroundColor;
uniform float yoffset;

void main()
{
    float x = texCoord.x;
    float y = texCoord.y + yoffset;

    //vec4 sample = texture2D(frequencyData, vec2(x, y));
    //float k = pow(sample.a, 2.0);
    //vec4 color = mix(vec4(0, 0, 0, 1), foregroundColor, k);

    //// Fade out the mesh close to the edges
    //float fade = pow(cos((1.0 - texCoord.y) * 0.5 * 3.1415926535), 0.5);
    //gl_FragColor = mix(vec4(0, 0, 0, 0), color, fade);

    vec4 sample = texture2D(frequencyData, vec2(x, y));
    vec4 ecGreen = vec4(0.164705, 0.521568, 0.305882, 1.0);
    vec4 ecRed = vec4(0.721568, 0.156862, 0.215686, 1.0);
    float k = pow(sample.a, 2.0);
    vec4 color = mix(ecGreen, ecRed, k);

    // Fade out the mesh close to the edges
    float fade = pow(cos((1.0 - texCoord.y) * 0.5 * 3.1415926535), 0.5);
    gl_FragColor = mix(vec4(0, 0, 0, 0), color, fade);
}
