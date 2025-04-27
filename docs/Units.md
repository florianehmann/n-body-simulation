# Units

We want to choose units such the actual numeric values in the simulation never stray to far from one. This way we can prevent excessive accumulation of floating point error. In a simulation ov gravity, this means we have to fix three dimensions: length, time, and mass. To do this, we can either intoduce constants for these three dimensions, or, what we're going to do here, introduce constants for length and mass and fix the gravitational constant to one: $G=1$. This means we have the dimensionful scales
$$l_0$$
for length,
$$m_0$$
for mass, and
$$
    t_0 = \sqrt{\frac{l_0^3}{G m_0}}
$$
for time.

From those, we can derive the energy scale
$$
    E_0 = \frac{G m_0^2}{l_0}
$$
and the angular momentum scale
$$
    L_0 = \sqrt{G l_0 m_0^3}\, .
$$

## Example: Galaxy

For a galaxy like the Milky Way, we could choose the following scales:
$$\begin{split}
    l_0 &= 1\,\mathrm{kpc} \approx 3.1 \cdot 10^{19}\,\mathrm{m} \quad\text{and}\\
    m_0 &= 10^{11}\,M_\odot \approx 2.0 \cdot 10^{41}\,\mathrm{kg} \, .
\end{split}$$

This would then imply the remaining scales:
$$\begin{split}
    t_0 &\approx 1.5\,\mathrm{Myr} \approx 4.7 \cdot 10^{13}\,\mathrm{s} \, ,\\
    E_0 &\approx 8.6 \cdot 10^{52} \, \mathrm{J} \, , \\
    L_0 &\approx 4.0 \cdot 10^{66} \, \mathrm{Nm} \, .
\end{split}$$

With these scales, we can model the Milky Way in a strongly simplified manner as a rotating Gaussian cloud with parameters
$$\begin{split}
    \sigma_x &\approx 13.4\,\mathrm{kpc} \, ,\\
    \sigma_y &\approx 13.4\,\mathrm{kpc} \, ,\\
    \sigma_z &\approx 1.3\,\mathrm{kpc} \, ,
\end{split}$$
and rotational period
$$
    T \approx 300\,\mathrm{Myr} \,.
$$

This would translate into the dimensionless parameters
$$\begin{split}
    \tilde{\sigma}_x &= \sigma_x / L_0 \approx 13.4 \, ,\\
    \tilde{\sigma}_y &= \sigma_y / L_0 \approx 13.4 \, ,\\
    \tilde{\sigma}_z &= \sigma_z / L_0 \approx 1.3 \, ,
\end{split}$$
and
$$
    \tilde{T} = T/t_0 \approx 201.2 \,.
$$
