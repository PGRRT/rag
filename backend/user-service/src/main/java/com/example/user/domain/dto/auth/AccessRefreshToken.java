package com.example.user.domain.dto.auth;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.springframework.http.ResponseCookie;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class AccessRefreshToken {
    private String accessToken;
    private ResponseCookie refreshToken;
}
