clear all
close all
clc

circle_range = 0:0.01:2*pi;
rotation = exp(1j*circle_range);

k = 0:1:5;
phased = 2*pi*k/6;
rd = sqrt(1+sin(pi/5));
d_circ_1 = rd*rotation;
d_circ_2 = 1/rd*rotation;
d_points_1 = rd .* exp(1j*phased);
d_points_2 = 1/rd .* exp(1j*phased);

rk = 0.7806;
phasek = [2.0971,  0.0056,  1.0659, -1.0467,  3.1443, -2.1024];
k_circ_1 = rk*rotation;
k_circ_2 = 1/rk*rotation;
k_points_1 = rk .* exp(1j*phasek);
k_points_2 = 1/rk .* exp(1j*phasek);

rm = 0.7293;
phasem = [-0.9800,  1.1301, -2.0362,  2.1951,  3.2044,  0.0767];
m_circ_1 = rm*rotation;
m_circ_2 = 1/rm*rotation;
m_points_1 = rm .* exp(1j*phasem);
m_points_2 = 1/rm .* exp(1j*phasem);

fig1 = figure(1)
c0 = plot(real(d_circ_1), imag(d_circ_1), "k")
hold on
plot(real(d_circ_2), imag(d_circ_2), "k")
plot(real(d_points_1), imag(d_points_1), "ksquare")
plot(real(d_points_2), imag(d_points_2), "ksquare")

c1 = plot(real(k_circ_1), imag(k_circ_1), "m--")
plot(real(k_circ_2), imag(k_circ_2), "m--")
plot(real(k_points_1), imag(k_points_1), "msquare")
plot(real(k_points_2), imag(k_points_2), "msquare")

c2 = plot(real(m_circ_1), imag(m_circ_1), "b:")
plot(real(m_circ_2), imag(m_circ_2), "b:")
plot(real(m_points_1), imag(m_points_1), "bsquare")
plot(real(m_points_2), imag(m_points_2), "bsquare")

legend([c0,c1,c2], {"DiZeT","KAN","MLP"}, location = "north west")
xlabel("Real")
ylabel("Imaginary")
xlim([-2 2])
ylim([-2 2])
grid on
hold off

saveas(fig1, "constellation.eps", eps)