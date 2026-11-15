export interface ProvinceTarget {
  id: string;
  name: string;
  center: [number, number];
  zoom: number;
}

export const PROVINCE_TARGETS: ProvinceTarget[] = [
  { id: 'arg', name: 'Argentina (Todo el país)', center: [-64.0, -37.5], zoom: 4.1 },
  { id: '06', name: 'Buenos Aires', center: [-60.1, -36.6], zoom: 5.8 },
  { id: '02', name: 'CABA', center: [-58.44, -34.61], zoom: 10.5 },
  { id: '10', name: 'Catamarca', center: [-67.0, -27.3], zoom: 6.4 },
  { id: '22', name: 'Chaco', center: [-60.7, -26.4], zoom: 6.5 },
  { id: '26', name: 'Chubut', center: [-68.5, -43.7], zoom: 6.1 },
  { id: '14', name: 'Córdoba', center: [-63.8, -32.1], zoom: 6.4 },
  { id: '18', name: 'Corrientes', center: [-57.8, -28.7], zoom: 6.7 },
  { id: '30', name: 'Entre Ríos', center: [-59.2, -32.0], zoom: 6.7 },
  { id: '34', name: 'Formosa', center: [-59.9, -24.9], zoom: 6.7 },
  { id: '38', name: 'Jujuy', center: [-65.7, -23.3], zoom: 7.0 },
  { id: '42', name: 'La Pampa', center: [-65.5, -37.1], zoom: 6.4 },
  { id: '46', name: 'La Rioja', center: [-67.2, -29.7], zoom: 6.6 },
  { id: '50', name: 'Mendoza', center: [-68.5, -34.6], zoom: 6.2 },
  { id: '54', name: 'Misiones', center: [-54.6, -26.9], zoom: 7.3 },
  { id: '58', name: 'Neuquén', center: [-70.1, -38.7], zoom: 6.5 },
  { id: '62', name: 'Río Negro', center: [-67.2, -40.3], zoom: 6.1 },
  { id: '66', name: 'Salta', center: [-64.8, -24.3], zoom: 6.2 },
  { id: '70', name: 'San Juan', center: [-68.9, -30.9], zoom: 6.5 },
  { id: '74', name: 'San Luis', center: [-66.0, -33.6], zoom: 6.8 },
  { id: '78', name: 'Santa Cruz', center: [-69.9, -48.8], zoom: 5.8 },
  { id: '82', name: 'Santa Fe', center: [-60.9, -30.7], zoom: 6.2 },
  { id: '86', name: 'Santiago del Estero', center: [-63.3, -27.8], zoom: 6.5 },
  { id: '94', name: 'Tierra del Fuego', center: [-67.7, -54.3], zoom: 6.8 },
  { id: '90', name: 'Tucumán', center: [-65.4, -27.0], zoom: 7.7 },
];
