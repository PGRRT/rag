// // src/redux/slices/authSlice.ts
// import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
// import { userApi } from '@/lib/api/userApi';
// import { User, AuthState, Credentials, RegisterData } from '@/types/user';

// const initialState: AuthState = {
//   user: null,
//   isAuthenticated: false,
//   isLoading: false,
//   error: null,
// };

// export const loginUser = createAsyncThunk(
//   'auth/login',
//   async (credentials: Credentials, { rejectWithValue }) => {
//     try {
//       console.log("Logging in with credentials:", credentials);
//       const response = await userApi.loginUserClient(credentials);
//       return response.data;
//     } catch (error: any) {
//       return rejectWithValue(error.response?.data?.message || 'Login failed');
//     }
//   }
// );

// export const registerUser = createAsyncThunk(
//   'auth/register',
//   async (userData: RegisterData, { rejectWithValue }) => {
//     try {
//       const response = await userApi.saveUserClient(userData);
//       return response.data;
//     } catch (error: any) {
//       return rejectWithValue(error.response?.data?.message || 'Registration failed');
//     }
//   }
// );

// export const refreshToken = createAsyncThunk(
//   'auth/refreshToken',
//   async (_, { rejectWithValue }) => {
//     try {
//       const response = await userApi.refreshUserClient();
//       return response.data;
//     } catch (error: any) {
//       return rejectWithValue(error.response?.data?.message || 'Token refresh failed');
//     }
//   }
// );

// export const getCurrentUser = createAsyncThunk(
//   'auth/getCurrentUser',
//   async (_, { rejectWithValue }) => {
//     try {
//       const response = await userApi.getUserClient();
//       return response.data;
//     } catch (error: any) {
//       return rejectWithValue(error.response?.data?.message || 'Failed to get user');
//     }
//   }
// );

// export const logoutUser = createAsyncThunk(
//   'auth/logout',
//   async (_, { rejectWithValue }) => {
//     try {
//       await userApi.logoutUserClient();
//       return true;
//     } catch (error: any) {
//       return true;
//     }
//   }
// );

// const authSlice = createSlice({
//   name: 'auth',
//   initialState,
//   reducers: {
//     clearError: (state) => {
//       state.error = null;
//     },
//     updateUser: (state, action: PayloadAction<Partial<User>>) => {
//       if (state.user) {
//         state.user = { ...state.user, ...action.payload };
//       }
//     },
//     resetAuth: () => initialState,
//   },
//   extraReducers: (builder) => {
//     // Login
//     builder
//       .addCase(loginUser.pending, (state) => {
//         state.isLoading = true;
//         state.error = null;
//       })
//       .addCase(loginUser.fulfilled, (state, action) => {
//         state.isLoading = false;
//         state.isAuthenticated = true;
//         state.user = action.payload.user || action.payload;
//         state.error = null;
//       })
//       .addCase(loginUser.rejected, (state, action) => {
//         state.isLoading = false;
//         state.isAuthenticated = false;
//         state.user = null;
//         state.error = action.payload as string;
//       });

//     // Register
//     builder
//       .addCase(registerUser.pending, (state) => {
//         state.isLoading = true;
//         state.error = null;
//       })
//       .addCase(registerUser.fulfilled, (state, action) => {
//         state.isLoading = false;
//         state.isAuthenticated = true;
//         state.user = action.payload.user || action.payload;
//         state.error = null;
//       })
//       .addCase(registerUser.rejected, (state, action) => {
//         state.isLoading = false;
//         state.error = action.payload as string;
//       });

//     // Get Current User
//     builder
//       .addCase(getCurrentUser.pending, (state) => {
//         state.isLoading = true;
//       })
//       .addCase(getCurrentUser.fulfilled, (state, action) => {
//         state.isLoading = false;
//         state.isAuthenticated = true;
//         state.user = action.payload.user || action.payload;
//         state.error = null;
//       })
//       .addCase(getCurrentUser.rejected, (state, action) => {
//         state.isLoading = false;
//         state.isAuthenticated = false;
//         state.user = null;
//         state.error = action.payload as string;
//       });

//     // Refresh Token
//     builder
//       .addCase(refreshToken.fulfilled, (state, action) => {
//         if (action.payload.user) {
//           state.user = action.payload.user;
//         }
//       })
//       .addCase(refreshToken.rejected, (state) => {
//         state.isAuthenticated = false;
//         state.user = null;
//       });

//     // Logout
//     builder.addCase(logoutUser.fulfilled, () => initialState);
//   },
// });

// export const { clearError, updateUser, resetAuth } = authSlice.actions;
// export default authSlice.reducer;
